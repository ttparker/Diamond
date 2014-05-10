#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/KroneckerProduct>
#include "GlobalHamiltonianParameters.h"

typedef Eigen::Matrix<double, d, d> MatrixDd;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    rmMatrixXd;

#ifdef realObservables
    #define obsId_d Matrix<double, d, d>::Identity()
    #define obsId(size) MatrixXd::Identity(size, size)
    #define re
    
    typedef Eigen::Matrix<double, d, d> obsMatrixD_t;
    typedef Eigen::MatrixXd obsMatrixX_t;
#endif

#ifdef complexObservables
    #define obsId_d Matrix<std::complex<double>, d, d>::Identity()
    #define obsId(size) MatrixXcd::Identity(size, size)
    #define re std::real
    
    typedef Eigen::Matrix<std::complex<double>, d, d> obsMatrixD_t;
    typedef Eigen::MatrixXcd obsMatrixX_t;
#endif

#endif
