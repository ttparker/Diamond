PROG = Diamond
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++11 -march=native -I ~/Eigen_3.2.0 $(DEBUG)
LIBS = -llapack
OBJS = FinalSuperblock.o ESolver.o FreeFunctions.o Lanczos.o main.o TheBlock.o $(PROG).o
COMMONHS1 = GlobalHamiltonianParameters.h main.h
COMMONHS2 = $(COMMONHS1) Hamiltonian.h TheBlock.h FinalSuperblock.h FreeFunctions.h ESolver.h
light = rm -f *.cpp~ *.h~ Makefile~
git = rm -f $(PROG) ./Output/*
deep = $(git) *.o

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBS) -o $(PROG) $(OBJS)

FinalSuperblock.o: $(COMMONHS2)

ESolver.o: $(COMMONHS1) Hamiltonian.h TheBlock.h ESolver.h

FreeFunctions.o: $(COMMONHS1) Hamiltonian.h TheBlock.h FinalSuperblock.h GlobalPrecisionParameters.h

Lanczos.o: $(COMMONHS1) ESolver.h GlobalPrecisionParameters.h

main.o: $(COMMONHS2) GlobalPrecisionParameters.h ObservableOps.h

TheBlock.o: $(COMMONHS2)

$(PROG).o: $(COMMONHS1) Hamiltonian.h

lightclean:
	$(light)

gitclean:
	$(git)

deepclean:
	$(deep)

clean:
	$(light)
	$(deep)

upload:
	scp -r *.cpp *.h Makefile $(OTHERS) knot.cnsi.ucsb.edu:~/$(DEST)

download:
	scp knot.cnsi.ucsb.edu:~/$(SOURCE)/Output/* Cluster
