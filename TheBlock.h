#ifndef THEBLOCK_H
#define THEBLOCK_H

#include "Hamiltonian.h"

class EffectiveHamiltonian;

class TheBlock
{
    public:
        int m;                              // number of states stored in block
        Eigen::MatrixXd primeToRhoBasis;              // change-of-basis matrix
        
        TheBlock(int m = 0,
                 const std::vector<int>& qNumList = std::vector<int>(),
                 const Eigen::MatrixXd& hS = Eigen::MatrixXd(),
                 const std::vector<Eigen::MatrixXd>& off0RhoBasisH2 
                        = std::vector<Eigen::MatrixXd>(),
                 const std::vector<Eigen::MatrixXd>& off1RhoBasisH2
                        = std::vector<Eigen::MatrixXd>(),
                 int l = 0);
        TheBlock(const Hamiltonian& ham, int mMaxIn);
        TheBlock nextBlock(rmMatrixXd& psiGround,
                           const TheBlock& compBlock, bool exactDiag = true,
                           bool infiniteStage = true,
                           const TheBlock& beforeCompBlock = TheBlock());
                                                     // performs each DMRG step
        EffectiveHamiltonian createHSuperFinal(const TheBlock& compBlock,
                                               const rmMatrixXd& psiGround,
                                               int skips) const;
                    // HSuperFinal, mSFinal, qNumList, oneSiteQNums, targetQNum
    
    private:
        std::vector<int> qNumList;
                // tracks the conserved quantum number of each row/column of hS
        Eigen::MatrixXd hS;                                // block Hamiltonian
        static Hamiltonian ham;
        std::vector<Eigen::MatrixXd> off0RhoBasisH2,
                                     off1RhoBasisH2;
            // density-matrix-basis coupling operators - "off" means the offset
            // between this block, in which the operator is represented, and
            // the site on which it acts
        int l;            // site at the end of the block (i.e. block size - 1)
        static int mMax;                   // max size of effective Hamiltonian
        
        Eigen::MatrixXd changeBasis(const Eigen::MatrixXd& mat) const;
                // represents operators in the basis of the new system block
    
    friend class EffectiveHamiltonian;
};

#endif
