#include <map>
#include <set>
#include "TheBlock.h"
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
    littleSeed /= littleSeed.norm();
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
                   int evecsToKeep)
{
    std::map<int, Sector> sectors;                 // key is the quantum number
    std::map<double, int> indexedEvals;              // eigenvalue, then sector
    std::set<int> qNumSet(qNumList.begin(), qNumList.end());
    for(int qNum : qNumSet)                 // make list of indexed eigenvalues
    {
        sectors.insert(sectors.end(),                          // create sector
                       std::make_pair(qNum, Sector(qNumList, qNum, mat)));
        sectors[qNum].solveForAll();
        for(int i = 0, end = sectors[qNum].multiplicity; i < end; i++)
        {
            double eval = sectors[qNum].solver.eigenvalues()(i);
            if(eval == 0.)                      // singular density matrix case
                eval = rand()%10000 * 1e-24;
            indexedEvals.insert(std::pair<double, int>(eval, qNum));
                                             // add indexed eigenvalues to list
        };
    };
    highestEvecQNums.reserve(evecsToKeep);
    int matSize = mat.rows();
    highestEvecs = MatrixX_t::Zero(matSize, evecsToKeep);
    auto currentIndexedEval = indexedEvals.rbegin();
    for(int j = 0; j < evecsToKeep; j++)
    {
        int qNum = currentIndexedEval++ -> second;
        highestEvecQNums.push_back(qNum);
        highestEvecs.col(j) = sectors[qNum].nextHighestEvec(matSize);
    };
};
