#include <map>
#include <set>
#include "TheBlock.h"
#include "ESolver.h"

using namespace Eigen;

double Sector::lancTolerance;
int Sector::fullMatrixSize;

void Sector::setLancTolerance(double newLancTolerance)
{
    lancTolerance = newLancTolerance;
};

Sector::Sector(const std::vector<int>& qNumList, int qNum, const MatrixXd& mat)
    : multiplicity(std::count(qNumList.begin(), qNumList.end(), qNum)),
      sectorMat(MatrixXd(multiplicity, multiplicity)),
      sectorColumnCounter(multiplicity)
{
    positions.reserve(multiplicity);
    for(auto firstElement = qNumList.begin(), qNumListElement = firstElement,
        end = qNumList.end(); qNumListElement != end; qNumListElement++)
        if(*qNumListElement == qNum)
            positions.push_back(qNumListElement - firstElement);
    int elmt = 0;
    for(int j : positions)			// fill sector matrix elements from matrix
        for(int i : positions)
            sectorMat(elmt++) = mat(i, j);
};

VectorXd Sector::filledOutEvec(VectorXd sectorEvec) const
{
    VectorXd longEvec = VectorXd::Zero(fullMatrixSize);
    for(int i = 0; i < multiplicity; i++)
        longEvec(positions[i]) = sectorEvec(i);
    return longEvec;
};

double Sector::solveForLowest(VectorXd& bigSeed)
{
    VectorXd littleSeed(multiplicity);
    for(int i = 0; i < multiplicity; i++)
        littleSeed(i) = bigSeed(positions[i]);
    littleSeed /= littleSeed.norm();
    double lowestEval = lanczos(sectorMat, littleSeed);
    bigSeed = filledOutEvec(littleSeed);
    return lowestEval;
};

void Sector::solveForAll()
{
    solver.compute(sectorMat);
};

Eigen::VectorXd Sector::nextHighestEvec()
{
    return filledOutEvec(solver.eigenvectors().col(--sectorColumnCounter));
};

HamSolver::HamSolver(const MatrixXd& mat, const std::vector<int>& qNumList,
                     int targetQNum, VectorXd& bigSeed)
    : storedLowestEvec(bigSeed)
{
    Sector::fullMatrixSize = mat.rows();
    Sector targetSector(qNumList, targetQNum, mat);
    storedLowestEval = targetSector.solveForLowest(storedLowestEvec);
};

VectorXd HamSolver::lowestEvec() const
{
    return storedLowestEvec;
};

double HamSolver::lowestEval() const
{
    return storedLowestEval;
};

DMSolver::DMSolver(const Eigen::MatrixXd& mat, const std::vector<int>& qNumList,
                   int evecsToKeep)
{
    int matSize = mat.rows();
    Sector::fullMatrixSize = matSize;
    std::map<int, Sector> sectors;			// key is the quantum number
    std::map<double, int> indexedEvals;		// eigenvalue, then sector
    std::set<int> qNumSet(qNumList.begin(), qNumList.end());
    for(int qNum : qNumSet)				// make list of indexed eigenvalues
    {
        sectors.insert(sectors.end(),					// create sector
                       std::make_pair(qNum, Sector(qNumList, qNum, mat)));
        sectors[qNum].solveForAll();
        for(int i = 0, end = sectors[qNum].multiplicity; i < end; i++)
            indexedEvals.insert(std::pair<double, int>
                                (sectors[qNum].solver.eigenvalues()(i), qNum));
                                            // add indexed eigenvalues to list
    };
    storedHighestEvecQNums.reserve(evecsToKeep);
    storedHighestEvecs = MatrixXd::Zero(matSize, evecsToKeep);
    auto currentIndexedEval = indexedEvals.rbegin();
    for(int j = 0; j < evecsToKeep; j++)
    {
        int qNum = currentIndexedEval++ -> second;
        storedHighestEvecQNums.push_back(qNum);
        storedHighestEvecs.col(j) = sectors[qNum].nextHighestEvec();
    };
};

MatrixXd DMSolver::highestEvecs() const
{
    return storedHighestEvecs;
};

std::vector<int> DMSolver::highestEvecQNums() const
{
    return storedHighestEvecQNums;
};
