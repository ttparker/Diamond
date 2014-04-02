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
#define siteType (thisBlock ? thisSiteType : compSiteType)

using namespace Eigen;

Hamiltonian::Hamiltonian() : thisSiteType(0), compSiteType(0),
                             oneSiteQNums({1, -1})
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
    BASJ << a,  c, a,
            0., b, b;
    LBRSJ = {b, b, 0.};
    LSRBJ = {b, 0., b};
    SSJ = {c, a, a};
    targetQNum = targetQNumIn;
    lSys = lSysIn;
};

MatrixXd Hamiltonian::blockAdjacentSiteJoin(int jType,
                                            const std::vector<MatrixXd>&
                                            rhoBasisH2, bool thisBlock) const
{
    MatrixXd plusMinus = kp(rhoBasisSigmaplus, sigmaminus);
    return BASJ(jType - 1, siteType) *
        (kp(rhoBasisSigmaz, sigmaz) + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixXd Hamiltonian::lBlockrSiteJoin(const std::vector<MatrixXd>&
                                      off0RhoBasisH2, int mlE, bool thisBlock)
                                      const
{
    MatrixXd plusMinus = kp(kp(off0RhoBasisSigmaplus, Id(d * mlE)), sigmaminus);
    return LBRSJ[siteType] * (kp(kp(off0RhoBasisSigmaz, Id(d * mlE)), sigmaz)
                              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixXd Hamiltonian::lSiterBlockJoin(int ml,
                                      const std::vector<MatrixXd>&
                                      off0RhoBasisH2, bool thisBlock) const
{
    MatrixXd plusMinus = kp(sigmaplus, off0RhoBasisSigmaplus.adjoint());
    return LSRBJ[siteType] *
        kp(kp(Id(ml), kp(sigmaz, off0RhoBasisSigmaz)
                      + 2 * (plusMinus + plusMinus.adjoint())),
           Id_d);
};

MatrixXd Hamiltonian::siteSiteJoin(int ml, int mlE, bool thisBlock) const
{
    MatrixXd plusMinus = kp(kp(sigmaplus, Id(mlE)), sigmaminus);
    return SSJ[siteType] * kp(Id(ml), kp(kp(sigmaz, Id(mlE)), sigmaz)
                                      + 2 * (plusMinus + plusMinus.adjoint()));
};
