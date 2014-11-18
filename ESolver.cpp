#include <map>
#include <set>
#include "GlobalPrecisionParameters.h"
#include "main.h"
#include "FreeFunctions.h"

using namespace Eigen;

std::vector<int> findTargetQNumPositions(const std::vector<int>& qNumList,
                                         int targetQNum)
{
    std::vector<int> positions; // which rows and columns of matrix are in sector
    for(auto firstElement = qNumList.begin(),
        qNumListElement = firstElement, end = qNumList.end();
        qNumListElement != end; qNumListElement++)
        if(*qNumListElement == targetQNum)
            positions.push_back(qNumListElement - firstElement);
    return positions;
};

MatrixX_t createCrossSectorBlock(const MatrixX_t& mat,
                                 const std::vector<int>& rowPositions,
                                 const std::vector<int>& colPositions)
                                         // pick out one block from full matrix
{
    MatrixX_t crossSectorBlock(rowPositions.size(), colPositions.size());
    int elmt = 0;
    for(int j : colPositions)
        for(int i : rowPositions)
            crossSectorBlock(elmt++) = mat(i, j);
    return crossSectorBlock;
};

VectorX_t filledOutEvec(const VectorX_t& sectorEvec, int fullMatrixSize,
                        const std::vector<int>& positions)
{
    VectorX_t longEvec = VectorX_t::Zero(fullMatrixSize);
    for(int i = 0, end = sectorEvec.size(); i < end; i++)
        longEvec(positions[i]) = sectorEvec(i);
    return longEvec;
};

HamSolver::HamSolver(const MatrixX_t& mat,
                     const std::vector<int>& hSprimeQNumList,
                     const std::vector<int>& hEprimeQNumList,
                     int targetQNum, rmMatrixX_t& bigSeed, double lancTolerance)
    : hSprimeQNumList(hSprimeQNumList), hEprimeQNumList(hEprimeQNumList),
      targetQNum(targetQNum)
{
    std::vector<int> positions
        = findTargetQNumPositions(vectorProductSum(hSprimeQNumList,
                                                   hEprimeQNumList),
                                  targetQNum);
    MatrixX_t sectorMat = createCrossSectorBlock(mat, positions, positions);
    int multiplicity = positions.size();
    rmMatrixX_t littleSeed(multiplicity, 1);
    for(int i = 0; i < multiplicity; i++)
        littleSeed(i) = bigSeed(positions[i]);
    littleSeed.normalize();
    lowestEval = lanczos(sectorMat, littleSeed, lancTolerance);
    lowestEvec = filledOutEvec(littleSeed, mat.rows(), positions);
};

DMSolver::DMSolver(const HamSolver hSuperSolver, int maxEvecsToKeep)
    : truncationError(0.)
{
    std::set<int> qNumSet(hSuperSolver.hSprimeQNumList.begin(),
                          hSuperSolver.hSprimeQNumList.end());
    std::map<int, std::vector<int>> qNumPositions;
    // lists positions of each quantum number within hSuperSolver.hSprimeQNumList
    std::multimap<double, std::pair<int, VectorX_t>> indexedEvecs;
                                // DM eigenvalue, then sector, then eigenvector
    for(int qNum : qNumSet)                 // make list of indexed eigenvalues
        if(std::count(hSuperSolver.hEprimeQNumList.begin(),
                      hSuperSolver.hEprimeQNumList.end(),
                      hSuperSolver.targetQNum - qNum))
        {
            std::vector<int> rowPositions
                = findTargetQNumPositions(hSuperSolver.hSprimeQNumList, qNum);
            qNumPositions
                .insert(qNumPositions.end(),
                        std::pair<int, std::vector<int>>(qNum, rowPositions));
            MatrixX_t crossSectorBlock
                = createCrossSectorBlock(hSuperSolver.lowestEvec,
                                         rowPositions,
                                         findTargetQNumPositions(hSuperSolver
                                                                 .hEprimeQNumList,
                                                                 hSuperSolver
                                                                 .targetQNum
                                                                 - qNum));
            SelfAdjointEigenSolver<MatrixX_t>
                DMSectorSolver(crossSectorBlock * crossSectorBlock.adjoint());
                                                 // find DM sector eigenvectors
            for(int i = 0, end = crossSectorBlock.rows(); i < end; i++)
                indexedEvecs
                    .insert(std::pair<double, std::pair<int, VectorX_t>>
                            (DMSectorSolver.eigenvalues()(i),
                             std::pair<int, VectorX_t>(qNum,
                                                       DMSectorSolver
                                                       .eigenvectors().col(i))));
                                             // add indexed eigenvalues to list
        };
    int matSize = hSuperSolver.lowestEvec.rows(),
        nIndexedEvecs = indexedEvecs.size(), // most states allowed by symmetry
        evecsToKeep = std::min({matSize, maxEvecsToKeep, nIndexedEvecs});
    auto weight = indexedEvecs.crbegin();
    std::advance(weight, evecsToKeep - 1);
    for(; evecsToKeep >= 1 && weight -> first == 0; evecsToKeep--, weight--);
                                // eliminate null vectors of the density matrix
    if(evecsToKeep < nIndexedEvecs)
                                 // have you truncated below the highest number
                                 // of DM eigenvectors allowed by symmetry?
    {
        for(; evecsToKeep >= 1
              && (weight -> first - std::next(weight, 1) -> first)
                 / std::abs(weight -> first) < degenerateDMCutoff;
            evecsToKeep--, weight--);
                   // find the highest-weighted eigenvector that does not terminate 
                   // inside a degenerate eigenspace of the density matrix
        weight++;                    // now points to first truncated DM eigenvalue
        for(auto end = indexedEvecs.crend(); weight != end; weight++)
            truncationError += weight -> first;
    };
    if(evecsToKeep == 0)
    {
        std::cerr << "More than mMax highest-weighted density-matrix "
                  << "eigenvectors are degenerate." << std::endl;
        exit(EXIT_FAILURE);
    }
    else if(evecsToKeep != maxEvecsToKeep)
        std::cout << "Warning: mMax truncation ends in a degenerate DM "
                  << "eigenspace, lowering cutoff to " << evecsToKeep
                  << " states." << std::endl;
    highestEvecQNums.reserve(evecsToKeep);
    highestEvecs = MatrixX_t::Zero(matSize, evecsToKeep);
    auto currentIndexedEvec = indexedEvecs.crbegin();
    for(int j = 0; j < evecsToKeep; j++, currentIndexedEvec++)
    {
        highestEvecQNums.push_back(currentIndexedEvec -> second.first);
        highestEvecs.col(j) = filledOutEvec(currentIndexedEvec -> second.second,
                                            matSize,
                                            qNumPositions[currentIndexedEvec
                                                          -> second.first]);
    };
};
