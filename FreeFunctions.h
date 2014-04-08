#ifndef FREEFUNCTIONS_H
#define FREEFUNCTIONS_H

#include "EffectiveHamiltonian.h"

std::vector<int> vectorProductSum(const std::vector<int>& first,
                                  const std::vector<int>& second);
            // takes the tensor product of two blocks' lists of quantum numbers
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
