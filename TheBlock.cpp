#include "FreeFunctions.h"

using namespace Eigen;

TheBlock::TheBlock(int m, const MatrixX_t& hS, const std::vector<int>& qNumList,
                   const matPair newRhoBasisH2s, int l)
    : m(m), hS(hS), siteType(l % nSiteTypes), qNumList(qNumList),
      off0RhoBasisH2(newRhoBasisH2s.first),
      off1RhoBasisH2(newRhoBasisH2s.second), l(l) {};

TheBlock::TheBlock(const Hamiltonian& ham)
    : m(d), hS(MatrixD_t::Zero()), siteType(0), qNumList(ham.oneSiteQNums), l(0)
{
    off0RhoBasisH2.assign(ham.siteBasisH2.begin(),
                          ham.siteBasisH2.begin() + nIndepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixX_t& psiGround,
                             double& cumulativeTruncationError)
{
    std::vector<int> hSprimeQNumList;
    MatrixX_t hSprime = createHprime(this, data.ham, hSprimeQNumList);
                                                       // expanded system block
    int md = m * d;
    if(data.exactDiag)
        return TheBlock(md, hSprime, hSprimeQNumList,
                        createNewRhoBasisH2s(data.ham.siteBasisH2, true), l + 1);
      // if near edge of system, no truncation necessary so skip DMRG algorithm
    HamSolver hSuperSolver = createHSuperSolver(data, hSprime, hSprimeQNumList,
                                                psiGround);
                                           // calculate superblock ground state
    int compm = data.compBlock -> m;
    hSuperSolver.lowestEvec.resize(md, compm * d);
    DMSolver rhoSolver(hSuperSolver, data.mMax);
                                             // find density matrix eigenstates
    cumulativeTruncationError += rhoSolver.truncationError;
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    int nextBlockm = primeToRhoBasis.cols();
    if(!data.infiniteStage) // modify psiGround to predict the next ground state
    {
        psiGround = hSuperSolver.lowestEvec;
        for(int sPrimeIndex = 0; sPrimeIndex < md; sPrimeIndex++)
                // transpose the environment block and the right-hand free site
        {
            rmMatrixX_t ePrime = psiGround.row(sPrimeIndex);
            ePrime.resize(compm, d);
            ePrime.transposeInPlace();
            ePrime.resize(1, d * compm);
            psiGround.row(sPrimeIndex) = ePrime;
        };
        psiGround = primeToRhoBasis.adjoint() * psiGround; 
                                      // change the expanded system block basis
        psiGround.resize(nextBlockm * d, compm);
        psiGround *= data.beforeCompBlock -> primeToRhoBasis.transpose();
                                          // change the environment block basis
        psiGround.resize(nextBlockm * d * data.beforeCompBlock -> m * d, 1);
    };
    return TheBlock(nextBlockm, changeBasis(hSprime), rhoSolver.highestEvecQNums,
                    createNewRhoBasisH2s(data.ham.siteBasisH2, false), l + 1);
                                  // save expanded-block operators in new basis
};

MatrixX_t TheBlock::createHprime(const TheBlock* block, const Hamiltonian& ham,
                                 std::vector<int>& hPrimeQNumList) const
{
    MatrixX_t hPrime = kp(block -> hS, Id_d)
                       + ham.blockAdjacentSiteJoin(1, block -> siteType,
                                                   block -> off0RhoBasisH2);
    if(l != 0)
        hPrime += ham.blockAdjacentSiteJoin(2, block -> siteType,
                                            block -> off1RhoBasisH2);
    hPrimeQNumList = vectorProductSum(block -> qNumList, ham.oneSiteQNums);
                                          // add in quantum numbers of new site
    return hPrime;
};

matPair TheBlock::createNewRhoBasisH2s(const vecMatD_t& siteBasisH2,
                                       bool exactDiag) const
{
    std::vector<MatrixX_t> newOff0RhoBasisH2,
                           newOff1RhoBasisH2;
    newOff0RhoBasisH2.reserve(nIndepCouplingOperators);
    newOff1RhoBasisH2.reserve(nIndepCouplingOperators);
    for(int i = 0; i < nIndepCouplingOperators; i++)
    {
        newOff0RhoBasisH2.push_back(exactDiag ?
                                    kp(Id(m), siteBasisH2[i]) :
                                    changeBasis(kp(Id(m), siteBasisH2[i])));
        newOff1RhoBasisH2.push_back(exactDiag ?
                                    kp(off0RhoBasisH2[i], Id_d) :
                                    changeBasis(kp(off0RhoBasisH2[i], Id_d)));
    };
    return std::make_pair(newOff0RhoBasisH2, newOff1RhoBasisH2);
};

HamSolver TheBlock::createHSuperSolver(const stepData& data,
                                       const MatrixX_t hSprime,
                                       const std::vector<int> hSprimeQNumList,
                                       rmMatrixX_t& psiGround) const
{
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
    int compm = data.compBlock -> m;
    MatrixX_t hSuper = kp(hSprime, Id(compm * d))
                       + data.ham.lBlockrSiteJoin(siteType, off0RhoBasisH2,
                                                  compm)
                       + data.ham.siteSiteJoin(siteType, m, compm)
                       + data.ham.lSiterBlockJoin(siteType, m,
                                                  data.compBlock
                                                  -> off0RhoBasisH2)
                       + kp(Id(m * d), hEprime);                  // superblock
    return HamSolver(hSuper, hSprimeQNumList, hEprimeQNumList, scaledTargetQNum,
                     psiGround, data.lancTolerance);
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
                                           // calculate superblock ground state
    return FinalSuperblock(hSuperSolver, data.ham.lSys, m, data.compBlock -> m,
                           skips);
};

obsMatrixX_t TheBlock::obsChangeBasis(const obsMatrixX_t& mat) const
{
    return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
