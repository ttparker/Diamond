#ifndef ESOLVER_H
#define ESOLVER_H

class Sector
{
    public:
        Sector() {};
    
    private:
        std::vector<int> positions;
                              // which rows and columns of matrix are in sector
        int multiplicity;                   // size of this symmetry sector
        Eigen::MatrixXd sectorMat;          // sector operator
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver; // DM eigensystem
        int sectorColumnCounter;            // tracks which sector eigenvector
                                            // to fill into a matrix eigenvector
        
        Sector(const std::vector<int>& qNumList, int qNum,
               const Eigen::MatrixXd& mat);
        Eigen::VectorXd filledOutEvec(Eigen::VectorXd sectorEvec,
                                      int fullMatrixSize) const;
        double solveForLowest(Eigen::VectorXd& lowestEvec, double lancTolerance),
               lanczos(const Eigen::MatrixXd& mat, rmMatrixXd& seed,
                       double lancTolerance);
     // changes input seed to ground eigenvector - make sure seed is normalized
        void solveForAll();
        Eigen::VectorXd nextHighestEvec(int fullMatrixSize);
    
    friend class HamSolver;
    friend class DMSolver;
};

class HamSolver
{
    public:
        Eigen::VectorXd lowestEvec;
        double lowestEval;
        
        HamSolver(const Eigen::MatrixXd& mat, const std::vector<int>& qNumList,
                  int targetQNum, rmMatrixXd& bigSeed, double lancTolerance);
};

class DMSolver
{
    public:
        Eigen::MatrixXd highestEvecs;
        std::vector<int> highestEvecQNums;
        
        DMSolver(const Eigen::MatrixXd& mat, const std::vector<int>& qNumList,
                 int evecsToKeep);
};

#endif
