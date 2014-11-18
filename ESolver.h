#ifndef ESOLVER_H
#define ESOLVER_H

class HamSolver
{
    public:
        rmMatrixX_t lowestEvec;
        double lowestEval;
        
        HamSolver(const MatrixX_t& mat, const std::vector<int>& hSprimeQNumList,
                  const std::vector<int>& hEprimeQNumList, int targetQNum,
                  rmMatrixX_t& bigSeed, double lancTolerance);
    
    private:
        std::vector<int> hSprimeQNumList,
                         hEprimeQNumList;
        int targetQNum;
        
        double lanczos(const MatrixX_t& mat, rmMatrixX_t& seed,
                       double lancTolerance);
     // changes input seed to ground eigenvector - make sure seed is normalized
    
    friend class DMSolver;
};

class DMSolver
{
    private:
        MatrixX_t highestEvecs;
        std::vector<int> highestEvecQNums;
        double truncationError;
        
        DMSolver(const HamSolver hSuperSolver, int maxEvecsToKeep);
    
    friend class TheBlock;
};

#endif
