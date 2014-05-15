#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/KroneckerProduct>
#include "GlobalHamiltonianParameters.h"

#ifdef realHamiltonian
    typedef Eigen::Matrix<double, d, d> MatrixD_t;
    typedef Eigen::MatrixXd MatrixX_t;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rmMatrixX_t;
    typedef Eigen::VectorXd VectorX_t;
#elif defined(complexHamiltonian)
    typedef Eigen::Matrix<std::complex<double>, d, d> MatrixD_t;
    typedef Eigen::MatrixXcd MatrixX_t;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> rmMatrixX_t;
    typedef Eigen::VectorXcd VectorX_t;
#endif

#if defined(realHamiltonian) && defined(realObservables)
    #define obsId_d Matrix<double, d, d>::Identity()
    #define obsId(size) MatrixXd::Identity(size, size)
    #define obsRe
    
    typedef Eigen::Matrix<double, d, d> obsMatrixD_t;
    typedef Eigen::MatrixXd obsMatrixX_t;
#elif defined(complexObservables) || defined(complexHamiltonian)
    // are their any complex elements in either the Hamiltonian or the observables?
    #define obsId_d Matrix<std::complex<double>, d, d>::Identity()
    #define obsId(size) MatrixXcd::Identity(size, size)
    #define obsRe std::real
    
    typedef Eigen::Matrix<std::complex<double>, d, d> obsMatrixD_t;
    typedef Eigen::MatrixXcd obsMatrixX_t;
#endif

#endif
