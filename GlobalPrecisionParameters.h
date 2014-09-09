#ifndef GPP_H
#define GPP_H

const int globalMinLancIters = 3,
          globalMaxLancIters = 100;
const double fallbackLancTolerance = 1.e-4,
              // reduced error tolerance to accept if Lanczos fails to converge
             observableThreshold = 1e-8;
// if any observable's absolute value is smaller than this, then round it to zero

#endif
