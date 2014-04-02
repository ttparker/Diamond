#include <fstream>
#include "EffectiveHamiltonian.h"
#include "GlobalPrecisionParameters.h"

using namespace Eigen;

std::vector<int> vectorProductSum(const std::vector<int>& first,
                                  const std::vector<int>& second)
{
    std::vector<int> prod;
    int firstSize = first.size(),
        secondSize = second.size();
    prod.reserve(firstSize * secondSize);
    for(int i = 0; i < firstSize; i++)
        for(int j = 0; j < secondSize; j++)
            prod.push_back(first.at(i) + second.at(j));
    return prod;
};

VectorXd oneSiteExpValues(const MatrixDd& oneSiteOp, int rangeOfObservables,
                          int currentLSys, EffectiveHamiltonian& hSuperFinal,
                          std::vector<TheBlock>& leftBlocks,
                          std::vector<TheBlock>& rightBlocks,
                          std::ofstream& fileout)
{
    opsVec ops;                     // list of observable single-site operators
    ops.push_back(std::make_pair(oneSiteOp, 0));
    VectorXd oneSiteVals(rangeOfObservables);
    int start = (currentLSys - rangeOfObservables) / 2;
    for(int i = 0; i < rangeOfObservables; i++)
    {
        ops[0].second = start + i;
        double exactValue = hSuperFinal.expValue(ops, leftBlocks, rightBlocks);
        oneSiteVals(i) = std::abs(exactValue) < observableThreshold ?
                         0. : exactValue;
    };
    fileout << "Expectation value of one-site observable at each site:\n"
            << oneSiteVals << std::endl << std::endl;
    return oneSiteVals;
};

MatrixXd twoSiteExpValues(const MatrixDd& firstTwoSiteOp,
                          const MatrixDd& secondTwoSiteOp, int rangeOfObservables,
                          int currentLSys, EffectiveHamiltonian& hSuperFinal,
                          std::vector<TheBlock>& leftBlocks,
                          std::vector<TheBlock>& rightBlocks,
                          std::ofstream& fileout)
{
    opsVec ops;                     // list of observable single-site operators
    ops.reserve(2);
    ops.push_back(std::make_pair(firstTwoSiteOp, 0));
    ops.push_back(std::make_pair(secondTwoSiteOp, 0));
    MatrixXd correlationFunction = ArrayXXd::Zero(rangeOfObservables,
                                                  rangeOfObservables);
    int start = (currentLSys - rangeOfObservables) / 2;
    for(int i = 0; i < rangeOfObservables; i++)
        for(int j = 0; j < rangeOfObservables; j++)
        {
            ops[0].second = start + i;
            ops[1].second = start + j;
            double exactValue = hSuperFinal.expValue(ops, leftBlocks,
                                                     rightBlocks);
            correlationFunction(i, j) = std::abs(exactValue)
                                            < observableThreshold ?
                                        0. : exactValue;
        };
    fileout << "Two-site correlation function:\n" << correlationFunction
            << std::endl << std::endl;
    return correlationFunction;
};
