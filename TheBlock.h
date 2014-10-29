#ifndef THEBLOCK_H
#define THEBLOCK_H

#include "Hamiltonian.h"
#include "ESolver.h"

typedef std::pair<std::vector<MatrixX_t>, std::vector<MatrixX_t>> matPair;

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
        MatrixX_t primeToRhoBasis;                    // change-of-basis matrix
        
        TheBlock(int m = 0,
                 const MatrixX_t& hS = MatrixX_t(),
                 const std::vector<int>& qNumList = std::vector<int>(),
                 const matPair newRhoBasisH2s = matPair(),
                 int l = 0);
        TheBlock(const Hamiltonian& ham);
        TheBlock nextBlock(const stepData& data, rmMatrixX_t& psiGround,
                           double& cumulativeTruncationError);
                                                     // performs each DMRG step
        FinalSuperblock createHSuperFinal(const stepData& data,
                                          rmMatrixX_t& psiGround, int skips)
                                          const;
                    // HSuperFinal, mSFinal, qNumList, oneSiteQNums, targetQNum
        obsMatrixX_t obsChangeBasis(const obsMatrixX_t& mat) const;
                       // changes basis during calculation of observables stage
    
    private:
        MatrixX_t hS;                                      // block Hamiltonian
        int siteType;                        // which site in the lattice basis
        std::vector<int> qNumList;
                // tracks the conserved quantum number of each row/column of hS
        std::vector<MatrixX_t> off0RhoBasisH2,
                               off1RhoBasisH2;
            // density-matrix-basis coupling operators - "off" means the offset
            // between this block, in which the operator is represented, and
            // the site on which it acts
        int l;            // site at the end of the block (i.e. block size - 1)
        
        MatrixX_t createHprime(const TheBlock* block, const Hamiltonian& ham,
                               std::vector<int>& hPrimeQNumList) const;
        matPair createNewRhoBasisH2s(const vecMatD_t& siteBasisH2,
                                     bool exactDiag) const;
        HamSolver createHSuperSolver(const stepData& data,
                                     const MatrixX_t hSprime,
                                     const std::vector<int> hSprimeQNumList,
                                     rmMatrixX_t& psiGround) const;
        MatrixX_t changeBasis(const MatrixX_t& mat) const;
                   // represents operators in the basis of the new system block
};

#endif
