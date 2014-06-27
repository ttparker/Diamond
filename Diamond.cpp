#include "Hamiltonian.h"

#define a couplingConstants[0]
#define b couplingConstants[1]
#define c couplingConstants[2]
#define sigmaplus h2[0]
#define sigmaz h2[1]
#define sigmaminus h2[2]
#define rhoBasisSigmaplus rhoBasisH2[0]
#define rhoBasisSigmaz rhoBasisH2[1]
#define off0RhoBasisSigmaplus off0RhoBasisH2[0]
#define off0RhoBasisSigmaz off0RhoBasisH2[1]
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
    BASJ <<  a, c, a,
            0., b, b;
    LBRSJ = {b, b, 0.};
    LSRBJ = {b, 0., b};
    SSJ = {c, a, a};
    targetQNum = targetQNumIn;
    lSys = lSysIn;
};

MatrixX_t Hamiltonian::blockAdjacentSiteJoin(int jType, int siteType,
                                             const std::vector<MatrixX_t>&
                                             rhoBasisH2) const
{
    MatrixX_t plusMinus = kp(rhoBasisSigmaplus, sigmaminus);
    return BASJ(jType - 1, siteType) *
           (kp(rhoBasisSigmaz, sigmaz) + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lBlockrSiteJoin(int siteType, const std::vector<MatrixX_t>&
                                       off0RhoBasisH2, int compm) const
{
    MatrixX_t plusMinus = kp(kp(off0RhoBasisSigmaplus, Id(d * compm)),
                             sigmaminus);
    return LBRSJ[siteType] * (kp(kp(off0RhoBasisSigmaz, Id(d * compm)), sigmaz)
                              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lSiterBlockJoin(int siteType, int m,
                                       const std::vector<MatrixX_t>&
                                       off0RhoBasisH2) const
{
    MatrixX_t plusMinus = kp(sigmaplus, off0RhoBasisSigmaplus.adjoint());
    return LSRBJ[siteType] *
        kp(kp(Id(m), kp(sigmaz, off0RhoBasisSigmaz)
                      + 2 * (plusMinus + plusMinus.adjoint())),
           Id_d);
};

MatrixX_t Hamiltonian::siteSiteJoin(int siteType, int m, int compm) const
{
    MatrixX_t plusMinus = kp(kp(sigmaplus, Id(compm)), sigmaminus);
    return SSJ[siteType] * kp(Id(m), kp(kp(sigmaz, Id(compm)), sigmaz)
                                     + 2 * (plusMinus + plusMinus.adjoint()));
};
