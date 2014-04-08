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

#endif
