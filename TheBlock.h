#ifndef THEBLOCK_H
#define THEBLOCK_H

#include "Hamiltonian.h"

class FinalSuperblock;
class TheBlock;

struct stepData
{
    Hamiltonian ham;                             // model Hamiltonian paramters
    bool exactDiag;             // close enough to edge to skip DMRG trucation?
    TheBlock* compBlock;         // complementary block on other side of system
    bool infiniteStage;
    double lancTolerance;  // max deviation from 1 of dot product of successive
                           // Lanczos iterations' ground state vectors
    int mMax;                              // max size of effective Hamiltonian
    TheBlock* beforeCompBlock;     // next smaller block than complementary one
};

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
        TheBlock(const Hamiltonian& ham);
        TheBlock nextBlock(const stepData& data, rmMatrixXd& psiGround);
                                                     // performs each DMRG step
        Eigen::MatrixXd changeBasis(const Eigen::MatrixXd& mat) const;
                   // represents operators in the basis of the new system block
        FinalSuperblock createHSuperFinal(const stepData& data,
                                          const rmMatrixXd& psiGround,
                                          int skips) const;
                    // HSuperFinal, mSFinal, qNumList, oneSiteQNums, targetQNum
    
    private:
        std::vector<int> qNumList;
                // tracks the conserved quantum number of each row/column of hS
        Eigen::MatrixXd hS;                                // block Hamiltonian
        std::vector<Eigen::MatrixXd> off0RhoBasisH2,
                                     off1RhoBasisH2;
            // density-matrix-basis coupling operators - "off" means the offset
            // between this block, in which the operator is represented, and
            // the site on which it acts
        int l;            // site at the end of the block (i.e. block size - 1)
};

#endif
