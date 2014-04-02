#ifndef ESOLVER_H
#define ESOLVER_H

class Sector
{
    public:
        Sector() {};
        static void setLancTolerance(double newLancTolerance);
        
    private:
        std::vector<int> positions;
                              // which rows and columns of matrix are in sector
        int multiplicity;                   // size of this symmetry sector
        Eigen::MatrixXd sectorMat;          // sector operator
        static double lancTolerance;
        static int fullMatrixSize;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver; // DM eigensystem
        int sectorColumnCounter;            // tracks which sector eigenvector
                                            // to fill into a matrix eigenvector
        
        Sector(const std::vector<int>& qNumList, int qNum,
               const Eigen::MatrixXd& mat);
        Eigen::VectorXd filledOutEvec(Eigen::VectorXd sectorEvec) const;
        double solveForLowest(Eigen::VectorXd& lowestEvec),
               lanczos(const Eigen::MatrixXd& mat, Eigen::VectorXd& seed);
     // changes input seed to ground eigenvector - make sure seed is normalized
        void solveForAll();
        Eigen::VectorXd nextHighestEvec();

    friend class HamSolver;
    friend class DMSolver;
};

class HamSolver
{
    public:
        HamSolver(const Eigen::MatrixXd& mat, const std::vector<int>& qNumList,
                  int targetQNum, Eigen::VectorXd& bigSeed);
        Eigen::VectorXd lowestEvec() const;
        double lowestEval() const;
    
    private:
        Eigen::VectorXd storedLowestEvec;
        double storedLowestEval;
};

class DMSolver
{
    public:
        DMSolver(const Eigen::MatrixXd& mat, const std::vector<int>& qNumList,
                 int evecsToKeep);
        Eigen::MatrixXd highestEvecs() const;
        std::vector<int> highestEvecQNums() const;
    
    private:
        Eigen::MatrixXd storedHighestEvecs;
        std::vector<int> storedHighestEvecQNums;
};

#endif
