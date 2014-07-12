#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "main.h"

#define kp kroneckerProduct
#define Id(size) MatrixXd::Identity(size, size)
#define Id_d Matrix<double, d, d>::Identity()   // one-site identity matrix

typedef std::vector<MatrixD_t, Eigen::aligned_allocator<MatrixD_t>> vecMatD_t;

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
        // listed below. The row gives the length of the coupling on the 
        // stretched-out chain, and the column gives the type of site within
        // the lattice basis.
        Eigen::Matrix<double, 5, nSiteTypes> LBRSJ,
                                             LSRBJ;
        std::vector<double> SSJ;
        vecMatD_t siteBasisH2;                 // site-basis coupling operators
        
        MatrixX_t blockAdjacentSiteJoin(int jType, int siteType,
                                        const std::vector<MatrixX_t>&
                                            offIRhoBasisH2)
                                        const,
                            // jType corresponds to the straightened-out chain,
                            // not the real-space coupling constants
                  lBlockrSiteJoin(int jType, int siteType,
                                  const std::vector<MatrixX_t>& offIRhoBasisH2,
                                  int compm) const,
                  lSiterBlockJoin(int jType, int siteType, int ml,
                                  const std::vector<MatrixX_t>& compOffIRhoBasisH2)
                                  const,
                  siteSiteJoin(int siteType, int ml, int compm) const,
                                           // joins the two free sites together
                  blockBlockJoin(int siteType, int l, int comp_l,
                                 const std::vector<std::vector<MatrixX_t>>&
                                     rhoBasisH2,
                                 const std::vector<std::vector<MatrixX_t>>&
                                     compRhoBasisH2) const,
                                               // joins the two blocks together
                  generalBlockBlockJoin(const std::vector<MatrixX_t>&
                                            offIRhoBasisH2,
                                        const std::vector<MatrixX_t>&
                                            compOffIRhoBasisH2) const;
    
    friend class TheBlock;
};

#endif
