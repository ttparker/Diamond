#include "FreeFunctions.h"
#include "ESolver.h"

using namespace Eigen;

EffectiveHamiltonian::EffectiveHamiltonian(const std::vector<int>& qNumList,
                                           const std::vector<int>& compQNumList,
                                           const Hamiltonian& ham,
                                           const MatrixXd& matFinal,
                                           int mSFinal, int skips)
    : lSupFinal(ham.lSys), mSFinal(mSFinal), skips(skips)
{
    VectorXd seed = TheBlock::psiGround;
    HamSolver hSuperSolver(matFinal,
                           vectorProductSum(vectorProductSum(qNumList,
                                                             ham.oneSiteQNums),
                                            vectorProductSum(compQNumList,
                                                             ham.oneSiteQNums)),
                           ham.targetQNum, seed);
    storedGSEnergy = hSuperSolver.lowestEval();
    psiGround = hSuperSolver.lowestEvec();
};

double EffectiveHamiltonian::gsEnergy() const
{
    return storedGSEnergy;
};

double EffectiveHamiltonian::expValue(const opsVec& ops,
                                      std::vector<TheBlock>& leftBlocks,
                                      std::vector<TheBlock>& rightBlocks)
{
    opsMap sysBlockOps, // observable operators that will act on the system block
           envBlockOps;                         // same for environment block
    int lSFinal = lSupFinal / 2 - 1;            // final size of system block
    MatrixDd lFreeSite = Id_d,
             rFreeSite = Id_d;
    bool opAtlFreeSite = false,
         opAtrFreeSite = false;
    for(const auto& opToEvaluate : ops) // split up operators into their blocks
    {
        int site = opToEvaluate.second;
        if(site < lSFinal)
            placeOp(opToEvaluate, sysBlockOps, true);
        else if(site == lSFinal)
        {                       // if necessary, assign left free-site operator
            lFreeSite *= opToEvaluate.first;
            opAtlFreeSite = true;
        }
        else if(site == lSFinal + 1)
        {                                       // and right free-site operator
            rFreeSite *= opToEvaluate.first;
            opAtrFreeSite = true;
        }
        else
            placeOp(opToEvaluate, envBlockOps, false);
    };
    if(sysBlockOps.empty() && !opAtlFreeSite)
                        // observables in right-hand half of superblock case
        if(envBlockOps.empty())
        {                   // right-hand free site single-site observable case
            psiGround.resize(mSFinal * d * mSFinal, d);
            return (psiGround
                    * rFreeSite.transpose()
                    * psiGround.adjoint()
                   ).trace();
        }
        else
        {
            psiGround.resize(mSFinal * d, mSFinal * d);
            return (psiGround.adjoint()
                    * psiGround
                    * kp(rhoBasisRep(envBlockOps, rightBlocks), rFreeSite).transpose()
                   ).trace();
        }
    else if(envBlockOps.empty() && !opAtrFreeSite)
                            // observables in left-hand half of superblock case
        if(!opAtlFreeSite)              // all observables in system block case
        {
            psiGround.resize(mSFinal, d * mSFinal * d);
            return (psiGround.adjoint()
                    * rhoBasisRep(sysBlockOps, leftBlocks)
                    * psiGround
                   ).trace();
        }
        else
        {
            psiGround.resize(mSFinal * d, mSFinal * d);
            return (psiGround.adjoint()
                    * kp(rhoBasisRep(sysBlockOps, leftBlocks), lFreeSite)
                    * psiGround
                   ).trace();
        }
    else                    // observables in both halves of superblock case
    {
    psiGround.resize(mSFinal * d, mSFinal * d);
    return (psiGround.adjoint()
            * kp(rhoBasisRep(sysBlockOps, leftBlocks), lFreeSite)
            * psiGround
            * kp(rhoBasisRep(envBlockOps, rightBlocks), rFreeSite).transpose()
           ).trace();
    };
};

void EffectiveHamiltonian::placeOp(const std::pair<MatrixDd, int>& op,
                                   opsMap& blockSide, bool systemSide)
{
    int lhSite = (systemSide ? op.second : lSupFinal - 1 - op.second);
    if(blockSide.count(lhSite))         // already an observable at this site?
        blockSide[lhSite] *= op.first;
    else
        blockSide.insert(std::pair<int, MatrixDd>(lhSite, op.first));
};

MatrixXd EffectiveHamiltonian::rhoBasisRep(const opsMap& blockOps,
                                           std::vector<TheBlock>& blocks) const
{
    if(blockOps.empty())
        return Id(blocks[lSupFinal / 2 - 2].m);
    MatrixXd rhoBasisBlockOp;
    const auto firstOp = blockOps.begin();
                // first nontrivial site operator at which to start tensoring
    auto currentOp = firstOp;
    const auto firstSite = blocks.begin();
    auto currentSite = firstSite;
    if(firstOp -> first == 0)       // one-site observable at first site?
    {
        rhoBasisBlockOp = firstOp -> second;
        currentOp++;
    }
    else
    {
        currentSite += (firstOp -> first - 1);
        rhoBasisBlockOp = Id(currentSite -> m);
    };
    const auto opsEnd = blockOps.end();
    for(const auto end = firstSite + lSupFinal / 2 - 2; currentSite != end;
        currentSite++)
    {
        MatrixXd primeBasisBlockOp = kp(rhoBasisBlockOp,
                                        (currentOp != opsEnd
                                         && currentSite - firstSite
                                            == currentOp -> first - 1 ?
                                        currentOp++ -> second :
                                        Id_d));
        rhoBasisBlockOp = (currentSite - firstSite < skips ?
                           primeBasisBlockOp :
                           currentSite -> changeBasis(primeBasisBlockOp));
    };
    return rhoBasisBlockOp;
};
