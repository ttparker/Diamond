#ifndef THEBLOCK_H
#define THEBLOCK_H

#include "Hamiltonian.h"
#include "ESolver.h"

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
    int mMax;                                             // max bond dimension
    TheBlock* beforeCompBlock;     // next smaller block than complementary one
};

class TheBlock
{
    public:
        int m;                              // number of states stored in block
        MatrixX_t primeToRhoBasis;                    // change-of-basis matrix
        
        TheBlock(int m = 0,
                 const std::vector<int>& qNumList = std::vector<int>(),
                 const MatrixX_t& hS = MatrixX_t(),
                 const std::vector<std::vector<MatrixX_t>>& rhoBasisH2 
                     = std::vector<std::vector<MatrixX_t>>(),
                 int l = 0);
        TheBlock(const Hamiltonian& ham);
        TheBlock nextBlock(const stepData& data, rmMatrixX_t& psiGround);
                                                     // performs each DMRG step
        FinalSuperblock createHSuperFinal(const stepData& data,
                                          rmMatrixX_t& psiGround, int skips)
                                          const;
        obsMatrixX_t obsChangeBasis(const obsMatrixX_t& mat) const;
                       // changes basis during calculation of observables stage
    
    private:
        std::vector<int> qNumList;
                // tracks the conserved quantum number of each row/column of hS
        MatrixX_t hS;                                      // block Hamiltonian
        std::vector<std::vector<MatrixX_t>> rhoBasisH2;
            // density-matrix-basis coupling operators - the first index gives
            // the offset between this block, in which the operator is
            // represented, and the site on which it acts.  The second index
            // gives the equivalent one-site operator
        int l;            // site at the end of the block (i.e. block size - 1)
        
        MatrixX_t createHprime(const TheBlock* block, const Hamiltonian& ham,
                               int siteType, std::vector<int>& hprimeQNumList)
                               const;
        std::vector<std::vector<MatrixX_t>>
            createNewRhoBasisH2(const vecMatD_t& siteBasisH2,
                                bool infiniteStage) const;
        HamSolver createHSuperSolver(const stepData& data,
                                     const MatrixX_t& hSprime,
                                     const std::vector<int>& hSprimeQNumList,
                                     int thisSiteType, rmMatrixX_t& psiGround)
                                     const;
        MatrixX_t changeBasis(const MatrixX_t& mat) const;
                   // represents operators in the basis of the new system block
};

#endif
