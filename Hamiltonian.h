#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "main.h"

#define kp kroneckerProduct
#define Id(size) MatrixXd::Identity(size, size)
#define Id_d Matrix<double, d, d>::Identity()   // one-site identity matrix

class Hamiltonian
{
    public:
        std::vector<int> oneSiteQNums;              // one-site quantum numbers
        int targetQNum,                              // targeted quantum number
            lSys;                                      // current system length
        
        Hamiltonian();
        void setParams(const std::vector<double>& couplingConstantsIn,
                       int targetQNumIn, int lSysIn);
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    private:
        std::vector<double> couplingConstants;
        Eigen::Matrix<double, 6, nSiteTypes> BASJ;
        // these are the arrays of coupling constants for each type of site in
        // the lattice basis - the acronyms stand for the coupling operators
        // listed below
        Eigen::Matrix<double, 5, nSiteTypes> LBRSJ,
                                             LSRBJ;
        std::vector<double> SSJ;
        std::vector<MatrixD_t, Eigen::aligned_allocator<MatrixD_t>> h2;
                                               // site-basis coupling operators
        
        MatrixX_t blockAdjacentSiteJoin(int jType, int siteType,
                                        const std::vector<MatrixX_t>& rhoBasisH2)
                                        const,
                            // jType corresponds to the straightened-out chain,
                            // not the real-space coupling constants
                  lBlockrSiteJoin(int jType, int siteType,
                                  const std::vector<MatrixX_t>& rhoBasisH2,
                                  int compm) const,
                  lSiterBlockJoin(int jType, int siteType, int ml,
                                  const std::vector<MatrixX_t>& compRhoBasisH2)
                                  const,
                  siteSiteJoin(int siteType, int ml, int compm) const,
                                           // joins the two free sites together
                  blockBlockJoin(int siteType, int l, int comp_l,
                                 const std::vector<MatrixX_t>& off0RhoBasisH2,
                                 const std::vector<MatrixX_t>& off1RhoBasisH2,
                                 const std::vector<MatrixX_t>& off2RhoBasisH2,
                                 const std::vector<MatrixX_t>& off3RhoBasisH2,
                                 const std::vector<MatrixX_t>&
                                     compOff0RhoBasisH2,
                                 const std::vector<MatrixX_t>&
                                     compOff1RhoBasisH2,
                                 const std::vector<MatrixX_t>&
                                     compOff2RhoBasisH2,
                                 const std::vector<MatrixX_t>&
                                     compOff3RhoBasisH2) const,
                                               // joins the two blocks together
                  generalBlockBlockJoin(const std::vector<MatrixX_t>&
                                            rhoBasisH2,
                                        const std::vector<MatrixX_t>&
                                            compRhoBasisH2) const;
    
    friend class TheBlock;
};

#endif
