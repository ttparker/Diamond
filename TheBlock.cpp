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
    rhoBasisH2.front().assign(ham.siteBasisH2.begin(),
                              ham.siteBasisH2.begin() + nIndepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixX_t& psiGround)
{
    std::vector<int> hSprimeQNumList;
    MatrixX_t hSprime = createHprime(this, data.ham, hSprimeQNumList);
                                                       // expanded system block
    if(data.exactDiag)
        return TheBlock(m * d, hSprimeQNumList, hSprime,
                        createNewRhoBasisH2(data.ham.siteBasisH2, true), l + 1);
      // if near edge of system, no truncation necessary so skip DMRG algorithm
    HamSolver hSuperSolver = createHSuperSolver(data, hSprime, hSprimeQNumList,
                                                psiGround);
                                           // calculate superblock ground state
    psiGround = hSuperSolver.lowestEvec;                        // ground state
    psiGround.resize(m * d, data.compBlock -> m * d);
    DMSolver rhoSolver(psiGround * psiGround.adjoint(), hSprimeQNumList,
                       data.mMax);           // find density matrix eigenstates
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    if(!data.infiniteStage) // modify psiGround to predict the next ground state
    {
        for(int sPrimeIndex = 0; sPrimeIndex < m * d; sPrimeIndex++)
                // transpose the environment block and the right-hand free site
        {
            rmMatrixX_t ePrime = psiGround.row(sPrimeIndex);
            ePrime.resize(data.compBlock -> m, d);
            ePrime.transposeInPlace();
            ePrime.resize(1, d * data.compBlock -> m);
            psiGround.row(sPrimeIndex) = ePrime;
        };
        psiGround = primeToRhoBasis.adjoint() * psiGround; 
                                      // change the expanded system block basis
        psiGround.resize(data.mMax * d, data.compBlock -> m);
        psiGround *= data.beforeCompBlock -> primeToRhoBasis.transpose();
                                          // change the environment block basis
        psiGround.resize(data.mMax * d * data.beforeCompBlock -> m * d, 1);
    };
    return TheBlock(data.mMax, rhoSolver.highestEvecQNums, changeBasis(hSprime),
                    createNewRhoBasisH2(data.ham.siteBasisH2, false), l + 1);
                                  // save expanded-block operators in new basis
};

MatrixX_t TheBlock::createHprime(const TheBlock* block, const Hamiltonian& ham,
                                 std::vector<int>& hprimeQNumList) const
{
    int siteType = block -> l % nSiteTypes;
         // calculate the position in the lattice basis of the end of the block
    MatrixX_t hprime = kp(block -> hS, Id_d);
    for(int i = 1; i <= farthestNeighborCoupling; i++)
        if(block -> l >= i - 1 && ham.BASJ(i - 1, siteType))
            hprime += ham.blockAdjacentSiteJoin(i, siteType,
                                                block -> rhoBasisH2[i - 1]);
                                                     // add in longer couplings
    hprimeQNumList = vectorProductSum(block -> qNumList, ham.oneSiteQNums);
                                          // add in quantum numbers of new site
    return hprime;
};

std::vector<std::vector<MatrixX_t>>
    TheBlock::createNewRhoBasisH2(const vecMatD_t& siteBasisH2, bool exactDiag)
    const
{
    std::vector<std::vector<MatrixX_t>> newRhoBasisH2(farthestNeighborCoupling);
    for(auto newOffIRhoBasisH2 : newRhoBasisH2)
        newOffIRhoBasisH2.reserve(nIndepCouplingOperators);
    for(int j = 0; j < nIndepCouplingOperators; j++)
    {
        newRhoBasisH2.front().push_back(exactDiag ?
                                        kp(Id(m), siteBasisH2[j]) :
                                        changeBasis(kp(Id(m), siteBasisH2[j])));
        for(int i = 0, end = farthestNeighborCoupling - 1; i < end; i++)
            if(l >= i)
                newRhoBasisH2[i + 1].push_back(exactDiag ?
                                               kp(rhoBasisH2[i][j], Id_d) :
                                               changeBasis(kp(rhoBasisH2[i][j],
                                                              Id_d)));
    };
    return newRhoBasisH2;
};

HamSolver TheBlock::createHSuperSolver(const stepData& data,
                                       const MatrixX_t& hSprime,
                                       const std::vector<int>& hSprimeQNumList,
                                       rmMatrixX_t& psiGround) const
{
    MatrixX_t hEprime;                            // expanded environment block
    std::vector<int> hEprimeQNumList;
    int scaledTargetQNum;
    if(data.infiniteStage)
    {
        hEprime = hSprime;
        hEprimeQNumList = hSprimeQNumList;
        scaledTargetQNum = data.ham.targetQNum * (l + 2) / data.ham.lSys * 2;
                                               // int automatically rounds down
                         // during iDMRG stage, target correct quantum number
                         // per unit site by scaling to fit current system size
                         // - note: this will change if d != 2
    }
    else
    {
        hEprime = createHprime(data.compBlock, data.ham,
                               hEprimeQNumList);
        scaledTargetQNum = data.ham.targetQNum;
    };
    int md = m * d,
        compm = data.compBlock -> m,
        compmd = compm * d,
        thisSiteType = l % nSiteTypes;
    MatrixX_t hlBlockrSite = MatrixX_t::Zero(md * compmd, md * compmd),
              hlSiterBlock = hlBlockrSite;
    if(data.infiniteStage && (data.ham.lSys - 2 * l - 4) % nSiteTypes)
                         // current system size incommensurate with final size?
    {
        int compSiteType = (data.ham.lSys - 4 - l) % nSiteTypes;
        for(int i = 2; i <= farthestNeighborCoupling; i++)
            if(l >= i - 2 && data.ham.LBRSJ(i - 2, thisSiteType))
            {
                hlBlockrSite += data.ham.lBlockrSiteJoin(i, thisSiteType,
                                                         rhoBasisH2[i - 2],
                                                         compm) / 2;
                hlSiterBlock
                    += data.ham.lSiterBlockJoin(i, compSiteType, m,
                                                data.compBlock
                                                -> rhoBasisH2[i - 2]) / 2;
            };
        for(int i = 2; i <= farthestNeighborCoupling; i++)
            if(data.compBlock -> l >= i - 2
               && data.ham.LSRBJ(i - 2, thisSiteType))
            {
                hlSiterBlock
                    += data.ham.lSiterBlockJoin(i, thisSiteType, m,
                                                data.compBlock
                                                -> rhoBasisH2[i - 2]) / 2;
                hlBlockrSite
                    += data.ham.lBlockrSiteJoin(i, compSiteType,
                                                rhoBasisH2[i - 2], compm) / 2;
            };
    }
    else
    {
        for(int i = 2; i <= farthestNeighborCoupling; i++)
            if(l >= i - 2 && data.ham.LBRSJ(i - 2, thisSiteType))
                hlBlockrSite += data.ham.lBlockrSiteJoin(i, thisSiteType,
                                                         rhoBasisH2[i - 2], compm);
        for(int i = 2; i <= farthestNeighborCoupling; i++)
            if(data.compBlock -> l >= i - 2
               && data.ham.LSRBJ(i - 2, thisSiteType))
                hlSiterBlock += data.ham.lSiterBlockJoin(i, thisSiteType, m,
                                                         data.compBlock
                                                         -> rhoBasisH2[i - 2]);
    };
    MatrixX_t hSuper = kp(hSprime, Id(compmd))
                          + hlBlockrSite
                          + data.ham.blockBlockJoin(thisSiteType, l,
                                                    data.compBlock -> l,
                                                    rhoBasisH2,
                                                    data.compBlock -> rhoBasisH2)
                          + hlSiterBlock
                          + kp(Id(md), hEprime);                  // superblock
    if(data.ham.SSJ[thisSiteType])
        hSuper += data.ham.siteSiteJoin(thisSiteType, m, compm);
    return HamSolver(hSuper, vectorProductSum(hSprimeQNumList, hEprimeQNumList),
                     scaledTargetQNum, psiGround, data.lancTolerance);
};

MatrixX_t TheBlock::changeBasis(const MatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};

FinalSuperblock TheBlock::createHSuperFinal(const stepData& data,
                                            rmMatrixX_t& psiGround, int skips)
                                            const
{
    std::vector<int> hSprimeQNumList;
    MatrixX_t hSprime = createHprime(this, data.ham, hSprimeQNumList);
                                                       // expanded system block
    HamSolver hSuperSolver = createHSuperSolver(data, hSprime, hSprimeQNumList,
                                                psiGround);
                                     // calculate final superblock ground state
    return FinalSuperblock(hSuperSolver, data.ham.lSys, m, data.compBlock-> m,
                           skips);
};

obsMatrixX_t TheBlock::obsChangeBasis(const obsMatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
