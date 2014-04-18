#include <cmath>
#include "main.h"
#include "ESolver.h"
#include "GlobalPrecisionParameters.h"

extern "C"
{
    void dstemr_(char* JOBZ, char* RANGE, int* N, double* D, double* E,
                 double* VL, double* VU, int* IL, int* IU, int* M, double* W,
                 double* Z, int* LDZ, int* NZC, int* ISUPPZ, bool* TRYRAC,
                 double* WORK, int* LWORK, int* IWORK, int* LIWORK, int* INFO);
};

using namespace Eigen;

double Sector::lanczos(const MatrixXd& mat, rmMatrixXd& seed,
                       double lancTolerance)
{
    int matSize = mat.rows();
    if(matSize == 1)
        return mat(0, 0);
    const int minIters = std::min(matSize, globalMinLancIters),
              maxIters = std::min(matSize, globalMaxLancIters);
    std::vector<double> a,
                        b;
    a.reserve(minIters);
    b.reserve(minIters);
    VectorXd x = seed;
    MatrixXd basisVecs = x;
    x.noalias() = mat * basisVecs;
    a.push_back(seed.col(0).dot(x));
    b.push_back(0.);
    VectorXd oldGS;
    int i = 0;                                      // iteration counter
    char JOBZ = 'V',                                // define dstemr arguments
         RANGE = 'I';
    int N = 1;
    std::vector<double> D,
                        E;
    D.reserve(minIters);
    E.reserve(minIters);
    double VL,
           VU;
    int IL = 1,
        IU = 1,
        M;
    std::vector<double> W;
    W.reserve(minIters);
    VectorXd Z;
    int LDZ,
        NZC = 1;
    std::vector<int> ISUPPZ;
    ISUPPZ.reserve(2);
    bool TRYRAC = true;
    double optLWORK;
    std::vector<double> WORK;
    int LWORK,
        optLIWORK;
    std::vector<int> IWORK;
    int LIWORK,
        INFO;
    do
    {
        i++;
        oldGS = seed;
        
        // Lanczos stage 1: Lanczos iteration
        x -= a[i - 1] * basisVecs.col(i - 1);
        b.push_back(x.norm());
        basisVecs.conservativeResize(NoChange, i + 1);
        basisVecs.col(i) = x / b[i];
        x.noalias() = mat * basisVecs.col(i) - b[i] * basisVecs.col(i - 1);
        a.push_back(basisVecs.col(i).dot(x));
        
        // Lanczos stage 2: diagonalize tridiagonal matrix
        N++;
        D = a;
        E.reserve(N);
        E.assign(b.begin() + 1, b.end());
        W.reserve(N);
        Z.resize(N);
        LDZ = N;
        LWORK = -1;
        LIWORK = -1;
        dstemr_(&JOBZ, &RANGE, &N, D.data(), E.data(), &VL, &VU, &IL, &IU, &M,
                W.data(), Z.data(), &LDZ, &NZC, ISUPPZ.data(), &TRYRAC,
                &optLWORK, &LWORK, &optLIWORK, &LIWORK, &INFO);
                                    // query for optimal workspace allocations
        LWORK = int(optLWORK);
        WORK.reserve(LWORK);
        LIWORK = optLIWORK;
        IWORK.reserve(LIWORK);
        dstemr_(&JOBZ, &RANGE, &N, D.data(), E.data(), &VL, &VU, &IL, &IU, &M,
                W.data(), Z.data(), &LDZ, &NZC, ISUPPZ.data(), &TRYRAC,
                WORK.data(), &LWORK, IWORK.data(), &LIWORK, &INFO);
        seed.noalias() = basisVecs * Z;
        seed /= seed.norm();
    } while(N < minIters ||
            (std::abs(1 - std::abs(seed.col(0).dot(oldGS))) > lancTolerance
             && N < maxIters));
    if(N == maxIters)
    {
        std::cerr << "Lanczos algorithm failed to converge after " << N
                  << " iterations." << std::endl;
        exit(EXIT_FAILURE);
    };
    return W.front();
};
