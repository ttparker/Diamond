#ifndef FREEFUNCTIONS_H
#define FREEFUNCTIONS_H

#include "EffectiveHamiltonian.h"

rmMatrixXd randomSeed(int size);            // outputs random normalized vector
std::vector<int> vectorProductSum(const std::vector<int>& first,
                                  const std::vector<int>& second);
            // takes the tensor product of two blocks' lists of quantum numbers
void reflectPredictedPsi(rmMatrixXd& psiGround, int mMax, int m);
                                               // when you reach edge of system
Eigen::VectorXd oneSiteExpValues(const MatrixDd& oneSiteOp,
                                 int rangeOfObservables, int lSys,
                                 EffectiveHamiltonian& hSuperFinal,
                                 std::vector<TheBlock>& leftBlocks,
                                 std::vector<TheBlock>& rightBlocks,
                                 std::ofstream& fileout);
Eigen::MatrixXd twoSiteExpValues(const MatrixDd& firstTwoSiteOp,
                                 const MatrixDd& secondTwoSiteOp,
                                 int rangeOfObservables, int lSys,
                                 EffectiveHamiltonian& hSuperFinal,
                                 std::vector<TheBlock>& leftBlocks,
                                 std::vector<TheBlock>& rightBlocks,
                                 std::ofstream& fileout);

#endif
