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
    off0RhoBasisH2.assign(ham.h2.begin(),
                          ham.h2.begin() + indepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixX_t& psiGround)
{
    int thisSiteType = l % nSiteTypes;
    MatrixX_t hSprime = kp(hS, Id_d)
                        + data.ham.blockAdjacentSiteJoin(1, thisSiteType,
                                                         off0RhoBasisH2);
                                                       // expanded system block
    if(l != 0)
        hSprime += data.ham.blockAdjacentSiteJoin(2, thisSiteType,
                                                  off1RhoBasisH2);
    std::vector<int> hSprimeQNumList = vectorProductSum(qNumList,
                                                        data.ham.oneSiteQNums);
                                          // add in quantum numbers of new site
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
            tempOff0RhoBasisH2.push_back(kp(Id(m), data.ham.h2[i]));
            tempOff1RhoBasisH2.push_back(kp(off0RhoBasisH2[i], Id_d));
        };
        return TheBlock(md, hSprimeQNumList, hSprime, tempOff0RhoBasisH2,
                        tempOff1RhoBasisH2, l + 1);
    };
    int compSiteType = data.compBlock -> l % nSiteTypes,
        compm = data.compBlock -> m,
        compmd = compm * d;
    MatrixX_t hEprime = (data.infiniteStage ?
                         hSprime :
                         kp(data.compBlock -> hS, Id_d)
                         + data.ham.blockAdjacentSiteJoin(1, compSiteType,
                                                          data.compBlock
                                                          -> off0RhoBasisH2)
                         + data.ham.blockAdjacentSiteJoin(2, compSiteType,
                                                          data.compBlock
                                                          -> off1RhoBasisH2));
                                                  // expanded environment block
    std::vector<int> hEprimeQNumList = (data.infiniteStage ?
                                        hSprimeQNumList :
                                        vectorProductSum(data.compBlock
                                                         -> qNumList,
                                                         data.ham.oneSiteQNums));
    int scaledTargetQNum = (data.infiniteStage ?
                            data.ham.targetQNum * (l + 2) / data.ham.lSys * 2 :
                                               // int automatically rounds down
                            data.ham.targetQNum);
                         // during iDMRG stage, targets correct quantum number
                         // per unit site by scaling to fit current system size
                         // - note: this will change if d != 2
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
        tempOff0RhoBasisH2.push_back(changeBasis(kp(Id(m), data.ham.h2[i])));
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

FinalSuperblock TheBlock::createHSuperFinal(const stepData& data,
                                            rmMatrixX_t& psiGround, int skips)
                                            const
{
    int thisSiteType = l % nSiteTypes;
    MatrixX_t hSprime = kp(hS, Id_d)
                        + data.ham.blockAdjacentSiteJoin(1, thisSiteType,
                                                         off0RhoBasisH2);
                                                       // expanded system block
    std::vector<int> hSprimeQNumList = vectorProductSum(qNumList,
                                                        data.ham.oneSiteQNums);
    if(l != 0)
        hSprime += data.ham.blockAdjacentSiteJoin(2, thisSiteType,
                                                  off1RhoBasisH2);
    int compSiteType = data.compBlock -> l % nSiteTypes,
        compm = data.compBlock -> m;
    MatrixX_t hEprime = kp(data.compBlock -> hS, Id_d)
                        + data.ham.blockAdjacentSiteJoin(1, compSiteType,
                                                         data.compBlock
                                                         -> off0RhoBasisH2)
                        + data.ham.blockAdjacentSiteJoin(2, compSiteType,
                                                         data.compBlock
                                                         -> off1RhoBasisH2);
                                                  // expanded environment block
    std::vector<int> hEprimeQNumList = vectorProductSum(data.compBlock
                                                        -> qNumList,
                                                        data.ham.oneSiteQNums);
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
