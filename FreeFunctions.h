#ifndef FREEFUNCTIONS_H
#define FREEFUNCTIONS_H

#include "FinalSuperblock.h"

rmMatrixXd randomSeed(const TheBlock& leftBlock, const TheBlock& rightBlock);
                                            // outputs random normalized vector
std::vector<int> vectorProductSum(const std::vector<int>& first,
                                  const std::vector<int>& second);
            // takes the tensor product of two blocks' lists of quantum numbers
void reflectPredictedPsi(rmMatrixXd& psiGround, const TheBlock& bigBlock,
                         const TheBlock& littleBlock);
                                               // when you reach edge of system
Eigen::VectorXd oneSiteExpValues(const obsMatrixD_t& oneSiteOp,
                                 int rangeOfObservables, int lSys,
                                 FinalSuperblock& hSuperFinal,
                                 std::vector<TheBlock>& leftBlocks,
                                 std::vector<TheBlock>& rightBlocks,
                                 std::ofstream& fileout);
Eigen::MatrixXd twoSiteExpValues(const obsMatrixD_t& firstTwoSiteOp,
                                 const obsMatrixD_t& secondTwoSiteOp,
                                 int rangeOfObservables, int lSys,
                                 FinalSuperblock& hSuperFinal,
                                 std::vector<TheBlock>& leftBlocks,
                                 std::vector<TheBlock>& rightBlocks,
                                 std::ofstream& fileout);

#endif
