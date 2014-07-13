#include "FreeFunctions.h"
#include "ESolver.h"

using namespace Eigen;

TheBlock::TheBlock(int m, const std::vector<int>& qNumList, const MatrixX_t& hS,
                   const std::vector<MatrixX_t>& off0RhoBasisH2,
                   const std::vector<MatrixX_t>& off1RhoBasisH2, int l)
    : m(m), qNumList(qNumList), hS(hS), off0RhoBasisH2(off0RhoBasisH2),
      off1RhoBasisH2(off1RhoBasisH2), l(l) {};

TheBlock::TheBlock(const Hamiltonian& ham)
    : m(d), qNumList(ham.oneSiteQNums), hS(MatrixD_t::Zero()), l(0)
{
    off0RhoBasisH2.assign(ham.siteBasisH2.begin(),
                          ham.siteBasisH2.begin() + indepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixX_t& psiGround)
{
    std::vector<int> hSprimeQNumList;
    MatrixX_t hSprime = createHprime(this, data.ham, hSprimeQNumList);
                                                       // expanded system block
    std::vector<MatrixX_t> tempOff0RhoBasisH2,
                           tempOff1RhoBasisH2;
    tempOff0RhoBasisH2.reserve(indepCouplingOperators);
    tempOff1RhoBasisH2.reserve(indepCouplingOperators);
    int md = m * d;
    if(data.exactDiag)
      // if near edge of system, no truncation necessary so skip DMRG algorithm
    {
        for(int i = 0; i < indepCouplingOperators; i++)
        {
            tempOff0RhoBasisH2.push_back(kp(Id(m), data.ham.siteBasisH2[i]));
            tempOff1RhoBasisH2.push_back(kp(off0RhoBasisH2[i], Id_d));
        };
        return TheBlock(md, hSprimeQNumList, hSprime, tempOff0RhoBasisH2,
                        tempOff1RhoBasisH2, l + 1);
    };
    MatrixX_t hEprime;
    std::vector<int> hEprimeQNumList;
    int scaledTargetQNum;
    if(data.infiniteStage)
    {
        hEprime = hSprime;
        hEprimeQNumList = hSprimeQNumList;
        scaledTargetQNum = data.ham.targetQNum * (l + 2) / data.ham.lSys * 2;
                                               // int automatically rounds down
                         // during iDMRG stage, targets correct quantum number
                         // per unit site by scaling to fit current system size
                         // - note: this will change if d != 2
    }
    else
    {
        hEprime = createHprime(data.compBlock, data.ham, hEprimeQNumList);
        scaledTargetQNum = data.ham.targetQNum;
    };
    int compm = data.compBlock -> m,
        compmd = compm * d,
        thisSiteType = l % nSiteTypes;
    HamSolver hSuperSolver(kp(hSprime, Id(compmd))
                           + data.ham.lBlockrSiteJoin(thisSiteType,
                                                      off0RhoBasisH2, compm)
                           + data.ham.siteSiteJoin(thisSiteType, m, compm)
                           + data.ham.lSiterBlockJoin(thisSiteType, m,
                                                      data.compBlock
                                                      -> off0RhoBasisH2)
                           + kp(Id(md), hEprime),
                           vectorProductSum(hSprimeQNumList, hEprimeQNumList),
                           scaledTargetQNum, psiGround, data.lancTolerance);
                                                 // find superblock eigenstates
    psiGround = hSuperSolver.lowestEvec;                        // ground state
    psiGround.resize(md, compmd);
    DMSolver rhoSolver(psiGround * psiGround.adjoint(), hSprimeQNumList,
                       data.mMax);           // find density matrix eigenstates
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    for(int i = 0; i < indepCouplingOperators; i++)
    {
        tempOff0RhoBasisH2.push_back(changeBasis(kp(Id(m),
                                                    data.ham.siteBasisH2[i])));
        tempOff1RhoBasisH2.push_back(changeBasis(kp(off0RhoBasisH2[i], Id_d)));
    };
    if(!data.infiniteStage) // modify psiGround to predict the next ground state
    {
        for(int sPrimeIndex = 0; sPrimeIndex < md; sPrimeIndex++)
                    // transpose the environment block and right-hand free site
        {
            rmMatrixX_t ePrime = psiGround.row(sPrimeIndex);
            ePrime.resize(compm, d);
            ePrime.transposeInPlace();
            ePrime.resize(1, compmd);
            psiGround.row(sPrimeIndex) = ePrime;
        };
        psiGround = primeToRhoBasis.adjoint() * psiGround; 
                                      // change the expanded system block basis
        psiGround.resize(data.mMax * d, compm);
        psiGround *= data.beforeCompBlock -> primeToRhoBasis.transpose();
                                          // change the environment block basis
        psiGround.resize(data.mMax * d
                         * data.beforeCompBlock -> primeToRhoBasis.rows(), 1);
    };
    return TheBlock(data.mMax, rhoSolver.highestEvecQNums, changeBasis(hSprime),
                    tempOff0RhoBasisH2, tempOff1RhoBasisH2, l + 1);
                                  // save expanded-block operators in new basis
};

MatrixX_t TheBlock::createHprime(const TheBlock* block, const Hamiltonian& ham,
                                 std::vector<int>& hPrimeQNumList) const
{
    int siteType = block -> l % nSiteTypes;
    MatrixX_t hPrime = kp(block -> hS, Id_d)
                       + ham.blockAdjacentSiteJoin(1, siteType,
                                                   block -> off0RhoBasisH2);
    if(l != 0)
        hPrime += ham.blockAdjacentSiteJoin(2, siteType,
                                            block -> off1RhoBasisH2);
    hPrimeQNumList = vectorProductSum(block -> qNumList, ham.oneSiteQNums);
                                          // add in quantum numbers of new site
    return hPrime;
};

FinalSuperblock TheBlock::createHSuperFinal(const stepData& data,
                                            rmMatrixX_t& psiGround, int skips)
                                            const
{
    std::vector<int> hSprimeQNumList;
    MatrixX_t hSprime = createHprime(this, data.ham, hSprimeQNumList);
                                                       // expanded system block
    std::vector<int> hEprimeQNumList;
    MatrixX_t hEprime = createHprime(data.compBlock, data.ham, hEprimeQNumList);
                                                  // expanded environment block
    int compm = data.compBlock -> m,
        thisSiteType = l % nSiteTypes;
    MatrixX_t hSuper = kp(hSprime, Id(compm * d))
                       + data.ham.lBlockrSiteJoin(thisSiteType, off0RhoBasisH2,
                                                  compm)
                       + data.ham.siteSiteJoin(thisSiteType, m, compm)
                       + data.ham.lSiterBlockJoin(thisSiteType, m,
                                                  data.compBlock
                                                  -> off0RhoBasisH2)
                       + kp(Id(m * d), hEprime);
    HamSolver hSuperSolver(hSuper,
                           vectorProductSum(hSprimeQNumList, hEprimeQNumList),
                           data.ham.targetQNum, psiGround, data.lancTolerance);
    return FinalSuperblock(hSuperSolver, data.ham.lSys, m, compm, skips);
};

obsMatrixX_t TheBlock::obsChangeBasis(const obsMatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};

MatrixX_t TheBlock::changeBasis(const MatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
