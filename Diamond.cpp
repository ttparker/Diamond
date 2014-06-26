#include "Hamiltonian.h"

#define jprime couplingConstants[0]
#define j1 couplingConstants[1]
#define j2 couplingConstants[2]
#define sigmaplus h2[0]
#define sigmaz h2[1]
#define sigmaminus h2[2]
#define rhoBasisSigmaplus rhoBasisH2[0]
#define rhoBasisSigmaz rhoBasisH2[1]
#define compRhoBasisSigmaplus compRhoBasisH2[0]
#define compRhoBasisSigmaz compRhoBasisH2[1]

using namespace Eigen;

Hamiltonian::Hamiltonian() : oneSiteQNums({1, -1})
{
    h2.resize(3);
    sigmaplus << 0., 1.,
                 0., 0.;
    sigmaminus << 0., 0.,
                  1., 0.;
    sigmaz << 1.,  0.,
              0., -1.;                                 // define Pauli matrices
};

void Hamiltonian::setParams(const std::vector<double>& couplingConstantsIn,
                            int targetQNumIn, int lSysIn)
{
    couplingConstants = couplingConstantsIn;
    BASJ << jprime,     0., jprime,
                0., jprime, jprime,
                j1,     j1,     0.,
                0.,     0.,     0.,
                0.,     0.,     0.,
                j2,     j2,     0.;
    LBRSJ << jprime, jprime, 0.,
                 j1,     0., j1,
                 0.,     0., 0.,
                 0.,     0., 0.,
                 j2,     0., j2;
    LSRBJ << jprime, 0., jprime,
                 j1, j1,     0.,
                 0., 0.,     0.,
                 0., 0.,     0.,
                 j2, j2,     0.;
    SSJ = {0., jprime, jprime};
    targetQNum = targetQNumIn;
    lSys = lSysIn;
};

MatrixX_t Hamiltonian::blockAdjacentSiteJoin(int jType, int siteType,
                                             const std::vector<MatrixX_t>&
                                             rhoBasisH2) const
{
    MatrixX_t plusMinus = kp(rhoBasisSigmaplus, sigmaminus);
    return BASJ(jType - 1, siteType)
           * (kp(rhoBasisSigmaz, sigmaz) + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lBlockrSiteJoin(int jType, int siteType,
                                       const std::vector<MatrixX_t>& rhoBasisH2,
                                       int compm) const
{
    MatrixX_t plusMinus = kp(kp(rhoBasisSigmaplus, Id(d * compm)), sigmaminus);
    return LBRSJ(jType - 2, siteType)
           * (kp(kp(rhoBasisSigmaz, Id(d * compm)), sigmaz)
              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lSiterBlockJoin(int jType, int siteType, int ml,
                                       const std::vector<MatrixX_t>&
                                       compRhoBasisH2) const
{
    MatrixX_t plusMinus = kp(sigmaplus, compRhoBasisSigmaplus.adjoint());
    return LSRBJ(jType - 2, siteType)
           * kp(kp(Id(ml), kp(sigmaz, compRhoBasisSigmaz)
                           + 2 * (plusMinus + plusMinus.adjoint())),
                Id_d);
};

MatrixX_t Hamiltonian::siteSiteJoin(int siteType, int ml, int compm) const
{
    MatrixX_t plusMinus = kp(kp(sigmaplus, Id(compm)), sigmaminus);
    return SSJ[siteType] * kp(Id(ml), kp(kp(sigmaz, Id(compm)), sigmaz)
                                      + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::blockBlockJoin(int siteType, int l, int comp_l,
                                      const std::vector<MatrixX_t>&
                                          off0RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          off1RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          off2RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          off3RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          compOff0RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          compOff1RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          compOff2RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          compOff3RhoBasisH2) const
{
    MatrixX_t bothBlocks;
    switch(siteType)
    {
        case 0:
        {
            int m = off0RhoBasisH2[1].rows(),
                compm = compOff0RhoBasisH2[1].rows();
            bothBlocks = MatrixX_t::Zero(m * d * compm * d, m * d * compm * d);
            if(l >= 2 && comp_l >= 1)
                bothBlocks += j2 * generalBlockBlockJoin(off2RhoBasisH2,
                                                         compOff1RhoBasisH2);
            if(l >= 1 && comp_l >= 2)
                bothBlocks += j2 * generalBlockBlockJoin(off1RhoBasisH2,
                                                         compOff2RhoBasisH2);
            break;
        }
        case 1:
            bothBlocks = j1 * generalBlockBlockJoin(off0RhoBasisH2,
                                                    compOff0RhoBasisH2);
            if(l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(off3RhoBasisH2,
                                                         compOff0RhoBasisH2);
            if(l >= 2 && comp_l >= 1)
                bothBlocks += j2 * generalBlockBlockJoin(off2RhoBasisH2,
                                                         compOff1RhoBasisH2);
            if(comp_l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(off0RhoBasisH2,
                                                         compOff3RhoBasisH2);
            break;
        case 2:
            bothBlocks = j1 * generalBlockBlockJoin(off0RhoBasisH2,
                                                    compOff0RhoBasisH2);
            if(l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(off3RhoBasisH2,
                                                         compOff0RhoBasisH2);
            if(l >= 1 && comp_l >= 2)
                bothBlocks += j2 * generalBlockBlockJoin(off1RhoBasisH2,
                                                         compOff2RhoBasisH2);
            if(comp_l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(off0RhoBasisH2,
                                                         compOff3RhoBasisH2);
            break;
    };
    return bothBlocks;
};

MatrixX_t Hamiltonian::generalBlockBlockJoin(const std::vector<MatrixX_t>&
                                                 rhoBasisH2,
                                             const std::vector<MatrixX_t>&
                                                 compRhoBasisH2) const
{
    MatrixX_t plusMinus = kp(kp(rhoBasisSigmaplus, Id_d),
                             kp(compRhoBasisSigmaplus.adjoint(), Id_d));
    return kp(kp(rhoBasisSigmaz, Id_d), kp(compRhoBasisSigmaz, Id_d))
           + 2 * (plusMinus + plusMinus.adjoint());
};
