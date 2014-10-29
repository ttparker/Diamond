#include <map>
#include <set>
#include "GlobalPrecisionParameters.h"
#include "main.h"
#include "ESolver.h"

using namespace Eigen;

Sector::Sector(const std::vector<int>& qNumList, int qNum, const MatrixX_t& mat)
    : multiplicity(std::count(qNumList.begin(), qNumList.end(), qNum)),
      sectorMat(MatrixX_t(multiplicity, multiplicity)),
      sectorColumnCounter(multiplicity)
{
    positions.reserve(multiplicity);
    for(auto firstElement = qNumList.begin(), qNumListElement = firstElement,
        end = qNumList.end(); qNumListElement != end; qNumListElement++)
        if(*qNumListElement == qNum)
            positions.push_back(qNumListElement - firstElement);
    int elmt = 0;
    for(int j : positions)           // fill sector matrix elements from matrix
        for(int i : positions)
            sectorMat(elmt++) = mat(i, j);
};

VectorX_t Sector::filledOutEvec(VectorX_t sectorEvec, int fullMatrixSize) const
{
    VectorX_t longEvec = VectorX_t::Zero(fullMatrixSize);
    for(int i = 0; i < multiplicity; i++)
        longEvec(positions[i]) = sectorEvec(i);
    return longEvec;
};

double Sector::solveForLowest(VectorX_t& bigSeed, double lancTolerance)
{
    rmMatrixX_t littleSeed(multiplicity, 1);
    for(int i = 0; i < multiplicity; i++)
        littleSeed(i) = bigSeed(positions[i]);
    littleSeed.normalize();
    double lowestEval = lanczos(sectorMat, littleSeed, lancTolerance);
    bigSeed = filledOutEvec(littleSeed, bigSeed.rows());
    return lowestEval;
};

void Sector::solveForAll()
{
    solver.compute(sectorMat);
};

VectorX_t Sector::nextHighestEvec(int fullMatrixSize)
{
    return filledOutEvec(solver.eigenvectors().col(--sectorColumnCounter),
                         fullMatrixSize);
};

HamSolver::HamSolver(const MatrixX_t& mat, const std::vector<int>& qNumList,
                     int targetQNum, rmMatrixX_t& bigSeed, double lancTolerance)
    : lowestEvec(bigSeed)
{
    Sector targetSector(qNumList, targetQNum, mat);
    lowestEval = targetSector.solveForLowest(lowestEvec, lancTolerance);
};

DMSolver::DMSolver(const MatrixX_t& mat, const std::vector<int>& qNumList,
                   int maxEvecsToKeep)
    : truncationError(0.)
{
    std::map<int, Sector> sectors;        // key is the sector's quantum number
    std::multimap<double, int> indexedEvals;         // eigenvalue, then sector
    std::set<int> qNumSet(qNumList.begin(), qNumList.end());
    for(int qNum : qNumSet)                 // make list of indexed eigenvalues
    {
        sectors.insert(sectors.end(),                          // create sector
                       std::make_pair(qNum, Sector(qNumList, qNum, mat)));
        sectors[qNum].solveForAll();
        for(int i = 0, end = sectors[qNum].multiplicity; i < end; i++)
            indexedEvals.insert(std::pair<double, int>(sectors[qNum].solver
                                                       .eigenvalues()(i), qNum));
                                             // add indexed eigenvalues to list
    };
    int matSize = mat.rows(),
        evecsToKeep;
    auto weight = indexedEvals.crbegin();
    if(matSize <= maxEvecsToKeep)
        evecsToKeep = matSize;
    else
    {
        evecsToKeep = maxEvecsToKeep;
        std::advance(weight, maxEvecsToKeep - 1);
        for(; evecsToKeep >= 1
              && (weight -> first == 0
                  || (weight -> first - std::next(weight, 1) -> first)
                      / std::abs(weight -> first) < degenerateDMCutoff);
            evecsToKeep--, weight--);
              // find the the max number of eigenvectors to keep that do not
              // terminate inside a degenerate eigenspace of the density matrix
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
    };
    weight++;                    // now points to first truncated DM eigenvalue
    for(auto end = indexedEvals.crend(); weight != end; weight++)
        truncationError += weight -> first;
    highestEvecQNums.reserve(evecsToKeep);
    highestEvecs = MatrixX_t::Zero(matSize, evecsToKeep);
    auto currentIndexedEval = indexedEvals.crbegin();
    for(int j = 0; j < evecsToKeep; j++)
    {
        int qNum = currentIndexedEval++ -> second;
        highestEvecQNums.push_back(qNum);
        highestEvecs.col(j) = sectors[qNum].nextHighestEvec(matSize);
    };
};
