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
    std::vector<int> hSprimeQNumList      // add in quantum numbers of new site
        = vectorProductSum(qNumList, data.ham.oneSiteQNums);
    int thisSiteType = l % nSiteTypes;
    MatrixX_t hSprime = kp(hS, Id_d)
                        + data.ham.blockAdjacentSiteJoin(1, thisSiteType,
                                                         off0RhoBasisH2);
                                                       // expanded system block
    if(l != 0)
        hSprime += data.ham.blockAdjacentSiteJoin(2, thisSiteType, off1RhoBasisH2);
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
    HamSolver hSuperSolver = (data.infiniteStage ? // find superblock eigenstates
                              HamSolver(MatrixX_t(kp(hSprime, Id(md))
                                                  + data.ham.lBlockrSiteJoin(thisSiteType, off0RhoBasisH2, m)
                                                  + data.ham.siteSiteJoin(thisSiteType, m, m)
                                                  + data.ham.lSiterBlockJoin(thisSiteType, m, off0RhoBasisH2)
                                                  + kp(Id(md), hSprime)),
                                        vectorProductSum(hSprimeQNumList,
                                                         hSprimeQNumList),
                                        data.ham.targetQNum * (l + 2) / data.ham.lSys * 2,
                                        psiGround, data.lancTolerance) :
                                               // int automatically rounds down
                              HamSolver(MatrixX_t(kp(hSprime, Id(compmd))
                                                  + data.ham.lBlockrSiteJoin(thisSiteType, off0RhoBasisH2, compm)
                                                  + data.ham.siteSiteJoin(thisSiteType, m, compm)
                                                  + data.ham.lSiterBlockJoin(thisSiteType, m, data.compBlock
                                                                                              -> off0RhoBasisH2)
                                                  + kp(Id(md), data.ham.blockAdjacentSiteJoin(1, compSiteType,
                                                                                              data.compBlock
                                                                                              -> off0RhoBasisH2)
                                                               + data.ham.blockAdjacentSiteJoin(2, compSiteType,
                                                                                                data.compBlock
                                                                                                -> off1RhoBasisH2)
                                                               + kp(data.compBlock -> hS, Id_d))),
                                        vectorProductSum(hSprimeQNumList,
                                                         vectorProductSum(data.compBlock -> qNumList,
                                                                          data.ham.oneSiteQNums)),
                                        data.ham.targetQNum, psiGround,
                                        data.lancTolerance));
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

obsMatrixX_t TheBlock::obsChangeBasis(const obsMatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};

FinalSuperblock TheBlock::createHSuperFinal(const stepData& data,
                                            const rmMatrixX_t& psiGround,
                                            int skips) const
{
    int thisSiteType = l % nSiteTypes,
        compSiteType = data.compBlock -> l % nSiteTypes,
        compm = data.compBlock -> m;
    return FinalSuperblock(MatrixX_t(kp(kp(hS, Id_d)
                                        + data.ham.blockAdjacentSiteJoin(1, thisSiteType,
                                                                         off0RhoBasisH2)
                                        + data.ham.blockAdjacentSiteJoin(2, thisSiteType,
                                                                         off1RhoBasisH2),
                                        Id(compm * d))
                                     + data.ham.lBlockrSiteJoin(thisSiteType,
                                                                off0RhoBasisH2,
                                                                compm)
                                     + data.ham.siteSiteJoin(thisSiteType, m,
                                                             compm)
                                     + data.ham.lSiterBlockJoin(thisSiteType, m,
                                                                data.compBlock
                                                                -> off0RhoBasisH2)
                                     + kp(Id(m * d), data.ham.blockAdjacentSiteJoin(1, compSiteType,
                                                                                    data.compBlock
                                                                                    -> off0RhoBasisH2)
                                                     + data.ham.blockAdjacentSiteJoin(2, compSiteType,
                                                                                      data.compBlock
                                                                                      -> off1RhoBasisH2)
                                                     + kp(data.compBlock -> hS, Id_d))),
                           qNumList, data.compBlock -> qNumList, data,
                           psiGround, m, compm, skips);
};

MatrixX_t TheBlock::changeBasis(const MatrixX_t& mat) const
{
	return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
