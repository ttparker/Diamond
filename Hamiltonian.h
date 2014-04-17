#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "main.h"

#define kp kroneckerProduct
#define Id(size) MatrixXd::Identity(size, size)
#define Id_d MatrixDd::Identity()   // one-site identity matrix

class Hamiltonian
{
    public:
        Hamiltonian();
        void setParams(const std::vector<double>& couplingConstants,
                       int targetQNumIn, int lSysIn);
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    private:
        std::vector<int> oneSiteQNums;
        Eigen::Matrix<double, 2, 3> BASJ;
        // these are the arrays of coupling constants for each type of site in
        // the lattice basis - the acronyms stand for the coupling operators
        // listed below
        std::vector<double> LBRSJ,
                            LSRBJ,
                            SSJ;
        std::vector<MatrixDd, Eigen::aligned_allocator<MatrixDd>> h2;
                                               // site-basis coupling operators
        int targetQNum,              // targeted average magnetization per site
            lSys;                                      // current system length
        
        Eigen::MatrixXd
            blockAdjacentSiteJoin(int jType, int siteType,
                                  const std::vector<Eigen::MatrixXd>& rhoBasisH2)
                                  const, // j gives the j-1th coupling constant
            lBlockrSiteJoin(int siteType,
                            const std::vector<Eigen::MatrixXd>& off0RhoBasisH2,
                            int mlE) const,
            lSiterBlockJoin(int siteType, int ml,
                            const std::vector<Eigen::MatrixXd>& off0RhoBasisH2)
                            const,
            siteSiteJoin(int siteType, int ml, int mlE) const;
                                           // joins the two free sites together
    
    friend class TheBlock;
    friend class EffectiveHamiltonian;
};

#endif
