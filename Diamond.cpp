#include "Hamiltonian.h"

#define jprime couplingConstants[0]
#define j1 couplingConstants[1]
#define sigmaplus h2[0]
#define sigmaz h2[1]
#define sigmaminus h2[2]
#define rhoBasisSigmaplus rhoBasisH2[0]
#define rhoBasisSigmaz rhoBasisH2[1]
#define off0RhoBasisSigmaplus off0RhoBasisH2[0]
#define off0RhoBasisSigmaz off0RhoBasisH2[1]
#define compOff0RhoBasisSigmaplus compOff0RhoBasisH2[0]
#define compOff0RhoBasisSigmaz compOff0RhoBasisH2[1]
#define off1RhoBasisSigmaplus off1RhoBasisH2[0]
#define off1RhoBasisSigmaz off1RhoBasisH2[1]

using namespace Eigen;

Hamiltonian::Hamiltonian() : oneSiteQNums({1, -1})
{
    h2.resize(3);
    sigmaplus << 0., 1.,
                 0., 0.;
    sigmaminus << 0., 0.,
                  1., 0.;
    sigmaz << 1., 0.,
              0., -1.;                                 // define Pauli matrices
};

void Hamiltonian::setParams(const std::vector<double>& couplingConstants,
                            int targetQNumIn, int lSysIn)
{
    BASJ << jprime, 0.,     jprime,
            0.,     jprime, jprime,
            j1,     j1,     0.;
    LBRSJ << jprime, jprime, 0.,
             j1,     0.,     j1;
    LSRBJ << jprime, 0., jprime,
             j1,     j1, 0.;
    SSJ = {0., jprime, jprime};
    BBJ = {0., j1, j1};
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
                                       int mlE) const
{
    MatrixX_t plusMinus = kp(kp(rhoBasisSigmaplus, Id(d * mlE)), sigmaminus);
    return LBRSJ(jType - 2, siteType)
           * (kp(kp(rhoBasisSigmaz, Id(d * mlE)), sigmaz)
              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lSiterBlockJoin(int jType, int siteType, int ml,
                                       const std::vector<MatrixX_t>&
                                       compOff0RhoBasisH2) const
{
    MatrixX_t plusMinus = kp(sigmaplus, compOff0RhoBasisSigmaplus.adjoint());
    return LSRBJ(jType - 2, siteType)
           * kp(kp(Id(ml), kp(sigmaz, compOff0RhoBasisSigmaz)
                           + 2 * (plusMinus + plusMinus.adjoint())),
                Id_d);
};

MatrixX_t Hamiltonian::siteSiteJoin(int siteType, int ml, int mlE) const
{
    MatrixX_t plusMinus = kp(kp(sigmaplus, Id(mlE)), sigmaminus);
    return SSJ[siteType] * kp(Id(ml), kp(kp(sigmaz, Id(mlE)), sigmaz)
                                      + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::blockBlockJoin(int siteType,
                                      const std::vector<MatrixX_t>&
                                          off0RhoBasisH2,
                                      const std::vector<MatrixX_t>&
                                          compOff0RhoBasisH2) const
{
    MatrixX_t plusMinus = kp(kp(off0RhoBasisSigmaplus, Id_d),
                             kp(compOff0RhoBasisSigmaplus.adjoint(), Id_d));
    return BBJ[siteType] * (kp(kp(off0RhoBasisSigmaz, Id_d),
                               kp(compOff0RhoBasisSigmaz, Id_d))
                            + 2 * (plusMinus + plusMinus.adjoint()));
};
