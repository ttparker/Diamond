#include "Hamiltonian.h"

#define jprime couplingConstants[0]
#define j1 couplingConstants[1]
#define j2 couplingConstants[2]
#define sigmaplus siteBasisH2[0]
#define sigmaz siteBasisH2[1]
#define sigmaminus siteBasisH2[2]
#define offIRhoBasisSigmaplus offIRhoBasisH2[0]
#define offIRhoBasisSigmaz offIRhoBasisH2[1]
#define compOffIRhoBasisSigmaplus compOffIRhoBasisH2[0]
#define compOffIRhoBasisSigmaz compOffIRhoBasisH2[1]

using namespace Eigen;

Hamiltonian::Hamiltonian() : oneSiteQNums({1, -1})
{
    siteBasisH2.resize(3);
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
                                             offIRhoBasisH2) const
{
    MatrixX_t plusMinus = kp(offIRhoBasisSigmaplus, sigmaminus);
    return BASJ(jType - 1, siteType)
           * (kp(offIRhoBasisSigmaz, sigmaz)
              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lBlockrSiteJoin(int jType, int siteType,
                                       const std::vector<MatrixX_t>&
                                           offIRhoBasisH2,
                                       int compm) const
{
    MatrixX_t plusMinus = kp(kp(offIRhoBasisSigmaplus, Id(d * compm)),
                             sigmaminus);
    return LBRSJ(jType - 2, siteType)
           * (kp(kp(offIRhoBasisSigmaz, Id(d * compm)), sigmaz)
              + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lSiterBlockJoin(int jType, int siteType, int m,
                                       const std::vector<MatrixX_t>&
                                       compOffIRhoBasisH2) const
{
    MatrixX_t plusMinus = kp(sigmaplus, compOffIRhoBasisSigmaplus.adjoint());
    return LSRBJ(jType - 2, siteType)
           * kp(kp(Id(m), kp(sigmaz, compOffIRhoBasisSigmaz)
                           + 2 * (plusMinus + plusMinus.adjoint())),
                Id_d);
};

MatrixX_t Hamiltonian::siteSiteJoin(int siteType, int m, int compm) const
{
    MatrixX_t plusMinus = kp(kp(sigmaplus, Id(compm)), sigmaminus);
    return SSJ[siteType] * kp(Id(m), kp(kp(sigmaz, Id(compm)), sigmaz)
                                      + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::blockBlockJoin(int siteType, int l, int comp_l,
                                      const std::vector<std::vector<MatrixX_t>>&
                                          rhoBasisH2,
                                      const std::vector<std::vector<MatrixX_t>>&
                                          compRhoBasisH2) const
{
    MatrixX_t bothBlocks;
    switch(siteType)
    {
        case 0:
        {
            int m = rhoBasisH2[0][1].rows(),
                compm = compRhoBasisH2[0][1].rows();
            bothBlocks = MatrixX_t::Zero(m * d * compm * d, m * d * compm * d);
            if(l >= 2 && comp_l >= 1)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[2],
                                                         compRhoBasisH2[1]);
            if(l >= 1 && comp_l >= 2)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[1],
                                                         compRhoBasisH2[2]);
            break;
        }
        case 1:
            bothBlocks = j1 * generalBlockBlockJoin(rhoBasisH2[0],
                                                    compRhoBasisH2[0]);
            if(l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[3],
                                                         compRhoBasisH2[0]);
            if(l >= 2 && comp_l >= 1)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[2],
                                                         compRhoBasisH2[1]);
            if(comp_l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[0],
                                                         compRhoBasisH2[3]);
            break;
        case 2:
            bothBlocks = j1 * generalBlockBlockJoin(rhoBasisH2[0],
                                                    compRhoBasisH2[0]);
            if(l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[3],
                                                         compRhoBasisH2[0]);
            if(l >= 1 && comp_l >= 2)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[1],
                                                         compRhoBasisH2[2]);
            if(comp_l >= 3)
                bothBlocks += j2 * generalBlockBlockJoin(rhoBasisH2[0],
                                                         compRhoBasisH2[3]);
            break;
    };
    return bothBlocks;
};

MatrixX_t Hamiltonian::generalBlockBlockJoin(const std::vector<MatrixX_t>&
                                                 offIRhoBasisH2,
                                             const std::vector<MatrixX_t>&
                                                 compOffIRhoBasisH2) const
{
    MatrixX_t plusMinus = kp(kp(offIRhoBasisSigmaplus, Id_d),
                             kp(compOffIRhoBasisSigmaplus.adjoint(), Id_d));
    return kp(kp(offIRhoBasisSigmaz, Id_d), kp(compOffIRhoBasisSigmaz, Id_d))
           + 2 * (plusMinus + plusMinus.adjoint());
};
