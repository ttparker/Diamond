#ifndef ESOLVER_H
#define ESOLVER_H

class Sector
{
    public:
        Sector() {};
    
    private:
        std::vector<int> positions;
                              // which rows and columns of matrix are in sector
        int multiplicity;                       // size of this symmetry sector
        MatrixX_t sectorMat;                                 // sector operator
        Eigen::SelfAdjointEigenSolver<MatrixX_t> solver;      // DM eigensystem
        int sectorColumnCounter;             // tracks which sector eigenvector
                                           // to fill into a matrix eigenvector
        
        Sector(const std::vector<int>& qNumList, int qNum, const MatrixX_t& mat);
        VectorX_t filledOutEvec(VectorX_t sectorEvec, int fullMatrixSize) const;
        double solveForLowest(VectorX_t& lowestEvec, double lancTolerance),
               lanczos(const MatrixX_t& mat, rmMatrixX_t& seed,
                       double lancTolerance);
     // changes input seed to ground eigenvector - make sure seed is normalized
        void solveForAll();
        VectorX_t nextHighestEvec(int fullMatrixSize);
    
    friend class HamSolver;
    friend class DMSolver;
};

class HamSolver
{
    public:
        VectorX_t lowestEvec;
        double lowestEval;
        
        HamSolver(const MatrixX_t& mat, const std::vector<int>& qNumList,
                  int targetQNum, rmMatrixX_t& bigSeed, double lancTolerance);
};

class DMSolver
{
    private:
        MatrixX_t highestEvecs;
        std::vector<int> highestEvecQNums;
        double truncationError;
        
        DMSolver(const MatrixX_t& mat, const std::vector<int>& qNumList,
                 int maxEvecsToKeep);
    
    friend class TheBlock;
};

#endif
