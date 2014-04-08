#ifndef EFFECTIVEHAMILTONIAN_H
#define EFFECTIVEHAMILTONIAN_H

#include <map>
#include "TheBlock.h"

typedef std::vector<std::pair<MatrixDd, int>,
                    Eigen::aligned_allocator<std::pair<MatrixDd, int>>> opsVec;
typedef std::map<int, MatrixDd, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, MatrixDd>>> opsMap;

class EffectiveHamiltonian
{
    public:
        double gsEnergy;                                 // ground-state energy
        
        EffectiveHamiltonian(const std::vector<int>& qNumList,
                             const std::vector<int>& compQNumList,
                             const Hamiltonian& ham,
                             const Eigen::MatrixXd& matFinal,
                             int mSFinal, int mEFinal, int skips);
        double expValue(const opsVec& ops, std::vector<TheBlock>& leftBlocks,
                        std::vector<TheBlock>& rightBlocks);
        // calculates exectation value of a combination of single-site operators
    
    private:
        int lSupFinal;                                     // final system size
        rmMatrixXd psiGround;                  // final superblock ground state
        int lSFinal,                            // final length of system block
            lEFinal,                       // final length of environment block
            mSFinal,           // final number of states stored in system block
            mEFinal,      // final number of states stored in environment block
            skips;                // number of edge sites in the position basis
        
        void placeOp(const std::pair<MatrixDd, int>& op, opsMap& blockSide,
                     bool systemSide);
                    // assign each one-site observable to the appropriate block
        Eigen::MatrixXd rhoBasisRep(const opsMap& blockOps,
                                    std::vector<TheBlock>& blocks,
                                    int blockSize) const;
                  // converts single-site operators into the system block basis
};

#endif
