#include "FreeFunctions.h"
#include "ESolver.h"

using namespace Eigen;

TheBlock::TheBlock(int m, const std::vector<int>& qNumList, const MatrixX_t& hS,
                   const std::vector<std::vector<MatrixX_t>>& rhoBasisH2, int l)
    : m(m), qNumList(qNumList), hS(hS), rhoBasisH2(rhoBasisH2), l(l) {};

TheBlock::TheBlock(const Hamiltonian& ham)
    : m(d), qNumList(ham.oneSiteQNums), hS(MatrixD_t::Zero()), l(0)
{
    rhoBasisH2.resize(farthestNeighborCoupling);
    rhoBasisH2.front().assign(ham.h2.begin(),
                              ham.h2.begin() + indepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixX_t& psiGround)
{
    int thisSiteType = l % nSiteTypes;
    MatrixX_t hSprime = kp(hS, Id_d);                  // expanded system block
    for(int i = 1; i <= farthestNeighborCoupling; i++)
        if(couplings[i] && l >= i - 1)
            hSprime += data.ham.blockAdjacentSiteJoin(i, thisSiteType,
                                                      rhoBasisH2[i - 1]);
                                                     // add in longer couplings
    std::vector<int> hSprimeQNumList = vectorProductSum(qNumList,
                                                        data.ham.oneSiteQNums);
                                          // add in quantum numbers of new site
    std::vector<std::vector<MatrixX_t>> tempRhoBasisH2(farthestNeighborCoupling);
    for(auto tempOffIRhoBasisH2 : tempRhoBasisH2)
        tempOffIRhoBasisH2.reserve(indepCouplingOperators);
    int md = m * d;
    if(data.exactDiag)
      // if near edge of system, no truncation necessary so skip DMRG algorithm
    {
        for(int j = 0; j < indepCouplingOperators; j++)
        {
            tempRhoBasisH2.front().push_back(kp(Id(m), data.ham.h2[j]));
            for(int i = 0, end = farthestNeighborCoupling - 1; i < end; i++)
                if(l >= i)
                    tempRhoBasisH2[i + 1].push_back(kp(rhoBasisH2[i][j], Id_d));
        };
        return TheBlock(md, hSprimeQNumList, hSprime, tempRhoBasisH2, l + 1);
    };
    int compSiteType = data.compBlock -> l % nSiteTypes,
        compm = data.compBlock -> m,
        compmd = compm * d,
        comp_l = (data.infiniteStage ? l : data.ham.lSys - l - 4);
    MatrixX_t hEprime;                            // expanded environment block
    if(data.infiniteStage)
        hEprime = hSprime;
    else
    {
        hEprime = kp(data.compBlock -> hS, Id_d);
        for(int i = 1; i <= farthestNeighborCoupling; i++)
            if(couplings[i] && comp_l >= i - 1)
                hEprime += data.ham.blockAdjacentSiteJoin(i, compSiteType,
                                                          data.compBlock
                                                          -> rhoBasisH2[i - 1]);
    };
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
                           + data.ham.lBlockrSiteJoin(2, thisSiteType,
                                                      rhoBasisH2[0], compm)
                           + data.ham.lBlockrSiteJoin(3, thisSiteType,
                                                      rhoBasisH2[1], compm)
                           + (l >= 4 ?
                              data.ham.lBlockrSiteJoin(6, thisSiteType,
                                                       rhoBasisH2[4], compm) :
                              MatrixX_t::Zero(md * compmd, md * compmd))
                           + data.ham.blockBlockJoin(thisSiteType, l, comp_l,
                                                     rhoBasisH2,
                                                     data.compBlock
                                                     -> rhoBasisH2)
                           + data.ham.siteSiteJoin(thisSiteType, m, compm)
                           + data.ham.lSiterBlockJoin(2, thisSiteType, m,
                                                      data.compBlock
                                                      -> rhoBasisH2[0])
                           + data.ham.lSiterBlockJoin(3, thisSiteType, m,
                                                      data.compBlock
                                                      -> rhoBasisH2[1])
                           + (comp_l >= 4 ?
                              data.ham.lSiterBlockJoin(6, thisSiteType, m,
                                                       data.compBlock
                                                       -> rhoBasisH2[4]) :
                              MatrixX_t::Zero(md * compmd, md * compmd))
                           + kp(Id(md), hEprime),
                           vectorProductSum(hSprimeQNumList, hEprimeQNumList),
                           scaledTargetQNum, psiGround, data.lancTolerance);
                                                 // find superblock eigenstates
    psiGround = hSuperSolver.lowestEvec;                        // ground state
    psiGround.resize(md, compmd);
    DMSolver rhoSolver(psiGround * psiGround.adjoint(), hSprimeQNumList,
                       data.mMax);           // find density matrix eigenstates
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    for(int j = 0; j < indepCouplingOperators; j++)
    {
        tempRhoBasisH2.front().push_back(changeBasis(kp(Id(m), data.ham.h2[j])));
        for(int i = 0, end = farthestNeighborCoupling - 1; i < end; i++)
            if(l >= i)
                tempRhoBasisH2[i + 1].push_back(changeBasis(kp(rhoBasisH2[i][j],
                                                               Id_d)));
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
                    tempRhoBasisH2, l + 1);
                                  // save expanded-block operators in new basis
};

FinalSuperblock TheBlock::createHSuperFinal(const stepData& data,
                                            const rmMatrixX_t& psiGround,
                                            int skips) const
{
    int thisSiteType = l % nSiteTypes,
        compSiteType = data.compBlock -> l % nSiteTypes,
        compm = data.compBlock -> m;
    MatrixX_t hSprime = kp(hS, Id_d);                  // expanded system block
    for(int i = 1; i <= farthestNeighborCoupling; i++)
        if(couplings[i] && l >= i - 1)
            hSprime += data.ham.blockAdjacentSiteJoin(i, thisSiteType,
                                                      rhoBasisH2[i - 1]);
    MatrixX_t hEprime = kp(data.compBlock -> hS, Id_d);
                                                  // expanded environment block
    for(int i = 1; i <= farthestNeighborCoupling; i++)
        if(couplings[i] && data.ham.lSys - l - 3 >= i)
            hEprime += data.ham.blockAdjacentSiteJoin(i, compSiteType,
                                                      data.compBlock
                                                      -> rhoBasisH2[i - 1]);
    return FinalSuperblock(kp(hSprime, Id(compm * d))
                           + data.ham.lBlockrSiteJoin(2, thisSiteType,
                                                      rhoBasisH2[0], compm)
                           + data.ham.lBlockrSiteJoin(3, thisSiteType,
                                                      rhoBasisH2[1], compm)
                           + (l >= 4 ?
                              data.ham.lBlockrSiteJoin(6, thisSiteType,
                                                       rhoBasisH2[4], compm) :
                              MatrixX_t::Zero(m * d * compm * d,
                                              m * d * compm * d))
                           + data.ham.blockBlockJoin(thisSiteType, l,
                                                     data.ham.lSys - l - 4,
                                                     rhoBasisH2,
                                                     data.compBlock
                                                     -> rhoBasisH2)
                           + data.ham.siteSiteJoin(thisSiteType, m, compm)
                           + data.ham.lSiterBlockJoin(2, thisSiteType, m,
                                                      data.compBlock
                                                      -> rhoBasisH2[0])
                           + data.ham.lSiterBlockJoin(3, thisSiteType, m,
                                                      data.compBlock
                                                      -> rhoBasisH2[1])
                           + (l < data.ham.lSys - 3 - 4 ?
                              data.ham.lSiterBlockJoin(6, thisSiteType, m,
                                                data.compBlock
                                                -> rhoBasisH2[4]) :
                              MatrixX_t::Zero(m * d * compm * d,
                                              m * d * compm * d))
                           + kp(Id(m * d), hEprime),
                           qNumList, data.compBlock -> qNumList, data,
                           psiGround, m, compm, skips);
};

obsMatrixX_t TheBlock::obsChangeBasis(const obsMatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};

MatrixX_t TheBlock::changeBasis(const MatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
