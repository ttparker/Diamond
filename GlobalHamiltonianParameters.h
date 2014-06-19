#ifndef GHP_H
#define GHP_H

const int d = 2,                              // size of one-site Hilbert space
          nCouplingConstants = 1,               // number of coupling constants
          nSiteTypes = 3,
              // number of distinct kinds of sites (e.g. size of Bravais basis)
          indepCouplingOperators = 2; // number of independent coupling operators

// only one of these next two lines should be uncommented, depending on whether
// the Hamiltonian has real or complex elements:
#define realHamiltonian
// #define complexHamiltonian

// only one of these next two lines should be uncommented, depending on whether
// the observable matrices have real or complex elements:
#define realObservables
// #define complexObservables

#endif
