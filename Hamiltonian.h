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
        void setParams(const std::vector<double>& couplingConstants,
                       int targetQNumIn, int lSysIn);
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    private:
        Eigen::Matrix<double, 2, 3> BASJ;
        // these are the arrays of coupling constants for each type of site in
        // the lattice basis - the acronyms stand for the coupling operators
        // listed below
        std::vector<double> LBRSJ,
                            LSRBJ,
                            SSJ;
        std::vector<MatrixD_t, Eigen::aligned_allocator<MatrixD_t>> h2;
                                               // site-basis coupling operators
        
        MatrixX_t blockAdjacentSiteJoin(int jType, int siteType,
                                        const std::vector<MatrixX_t>& rhoBasisH2)
                                        const,
                                         // j gives the j-1th coupling constant
                  lBlockrSiteJoin(int siteType,
                                  const std::vector<MatrixX_t>& off0RhoBasisH2,
                                  int mlE) const,
                  lSiterBlockJoin(int siteType, int ml,
                                  const std::vector<MatrixX_t>& off0RhoBasisH2)
                                  const,
                  siteSiteJoin(int siteType, int ml, int mlE) const;
                                           // joins the two free sites together
    
    friend class TheBlock;
};

#endif
