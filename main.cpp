#include <time.h>
#include <fstream>
#include "FreeFunctions.h"
#include "ESolver.h"
#include "GlobalPrecisionParameters.h"

using namespace Eigen;

int main()
{
    clock_t start = clock();

    std::ifstream filein("Input/Input");
    if(!filein)
    {
        std::cerr << "Couldn't open input file." << std::endl;
        exit(EXIT_FAILURE);
    };
    
    // read in parameters that are constant across all trials:
    int nTrials;
    bool calcObservables;
    filein >> nTrials >> calcObservables;
    bool calcOneSiteExpValues;
    int indexOfOneSiteOp;
    MatrixDd oneSiteOp;
    bool calcTwoSiteExpValues;
    int indexOfFirstTwoSiteOp,
        indexOfSecondTwoSiteOp;
    MatrixDd firstTwoSiteOp,
             secondTwoSiteOp;
    #include "ObservableOps.h"
    if(calcObservables)
    {
        filein >> calcOneSiteExpValues;
        if(calcOneSiteExpValues)
        {
            filein >> indexOfOneSiteOp;
            oneSiteOp = obsList[indexOfOneSiteOp];
        };
        filein >> calcTwoSiteExpValues;
        if(calcTwoSiteExpValues)
        {
            filein >> indexOfFirstTwoSiteOp >> indexOfSecondTwoSiteOp;
            firstTwoSiteOp = obsList[indexOfFirstTwoSiteOp];
            secondTwoSiteOp = obsList[indexOfSecondTwoSiteOp];
        };
    };
    
    std::ofstream infoout("Output/Info");
    if(infoout)
        infoout << "Number of trials: " << nTrials
                << "\nCalculate one-site observables? "
                << (calcOneSiteExpValues ? "Yes" : "No")
                << "\nIndex of one-site observable: " << indexOfOneSiteOp
                << "\nCalculate two-site observables? "
                << (calcTwoSiteExpValues ? "Yes" : "No")
                << "\nIndices of two-site observables: " << indexOfFirstTwoSiteOp
                << " " << indexOfSecondTwoSiteOp << "\nObservables threshold: "
                << observableThreshold << std::endl;
    else
    {
        std::cerr << "Couldn't open output files." << std::endl;
        exit(EXIT_FAILURE);
    };
    infoout.close();
    Hamiltonian ham;            // initialize the system's Hamiltonian
    for(int trial = 1; trial <= nTrials; trial++)
    {
        clock_t startTrial = clock();
        std::cout << "Trial " << trial << ":" << std::endl;
        std::ofstream fileout("Output/Trial_" + std::to_string(trial));
        fileout << "Trial " << trial << ":\n" << std::endl;
        
        // read in parameters that vary over trials:
        int lSys;                           // system length
        filein >> lSys;
        std::vector<double> couplingConstants(nCouplingConstants);
        for(int i = 0; i < nCouplingConstants; i++)
            filein >> couplingConstants[i];
        int targetQNum,
            rangeOfObservables;
        double groundStateErrorTolerance;
        int mMax,                           // max number of stored states
            nSweeps;                        // number of sweeps to be performed
        filein >> targetQNum >> rangeOfObservables >> groundStateErrorTolerance
               >> mMax >> nSweeps;
        if(rangeOfObservables == -1)
            rangeOfObservables = lSys;
        
        fileout << "System length: " << lSys << "\nCoupling constants:";
        for(int i = 0; i < nCouplingConstants; i++)
            fileout << " " << couplingConstants[i];
         fileout << "\nTargeted quantum number: " << targetQNum
                 << "\nLanczos tolerance: " << groundStateErrorTolerance
                 << "\nBond dimension: " << mMax << "\nNumber of sweeps: "
                 << nSweeps << std::endl << std::endl;
        ham.setParams(couplingConstants, targetQNum, lSys);
        int skips = 0,
            runningKeptStates = d * d;
        for(; runningKeptStates <= mMax; skips++)
            runningKeptStates *= d;  // find how many edge sites can be skipped
        bool oddSize = lSys % 2;
        int lSFinal,                        // final length of the system block
            lEFinal;                   // final length of the environment block
        if(oddSize)
        {
            lSFinal = (lSys - 1)/2;
            lEFinal = (lSys - 3)/2;
        }
        else
            lSFinal = lEFinal = lSys / 2 - 1;
        bool completeED = false;
        if(skips + 1 >= lSFinal)
        {
            if(skips + 1 == lSFinal && runningKeptStates == mMax * d)
            {
                std::cout << "Note: the bond dimension is large enough to "
                          << "perform exact diagonalization." << std::endl;
                completeED = true;
            }
            else
            {
                std::cout << "Error: the bond dimension is larger than "
                          << "required for exact diagonalization."
                          << std::endl;
                continue;
            };
        };
        std::vector<TheBlock> leftBlocks(lSys - 2 - skips),
                              rightBlocks(lSys - 2 - skips);
             // initialize system - the last block is only used for odd-size ED
        leftBlocks[0] = rightBlocks[0]
                      = TheBlock(ham, mMax);   // initialize the one-site block
        std::cout << "Performing iDMRG..." << std::endl;
            // note: this iDMRG code assumes parity symmetry of the Hamiltonian
        rmMatrixXd psiGround;                     // seed for Lanczos algorithm
        for(int site = 0; site < skips; site++)                   // initial ED
            rightBlocks[site + 1] = leftBlocks[site + 1]
                                  = leftBlocks[site].nextBlock(psiGround,
                                                               TheBlock());
        Sector::lancTolerance = groundStateErrorTolerance
                                * groundStateErrorTolerance / 2;
        for(int site = skips, end = lEFinal - 1; site < end; site++)   // iDMRG
        {
            psiGround = randomSeed(leftBlocks[site].m * d
                                   * leftBlocks[site].m * d);
            rightBlocks[site + 1] = leftBlocks[site + 1]
                                  = leftBlocks[site].nextBlock(psiGround,
                                                               TheBlock(),
                                                               false);
            rightBlocks[site].primeToRhoBasis = leftBlocks[site].primeToRhoBasis;
                                     // copy primeToRhoBasis to reflected block
        };
        if(oddSize)          // last half-step of iDMRG for an odd-sized system
        {
            psiGround = randomSeed(leftBlocks[lSFinal - 2].m * d
                                   * leftBlocks[lSFinal - 2].m * d);
            leftBlocks[lSFinal - 1] = leftBlocks[lSFinal - 2]
                                      .nextBlock(psiGround, TheBlock(),
                                                 completeED);
        };
        if(nSweeps == 0 || completeED)
            psiGround = randomSeed(leftBlocks[lSFinal - 1].m * d
                                   * rightBlocks[lEFinal - 1].m * d);
        else
        {
            std::cout << "Performing fDMRG..." << std::endl;
            psiGround = randomSeed(leftBlocks[lSFinal - 1].m * d
                                   * rightBlocks[lEFinal - 1].m * d);
            for(int i = 1; i <= nSweeps; i++)       // perform the fDMRG sweeps
            {
                for(int site = lSFinal - 1, end = lSys - 4 - skips; site < end;
                    site++)
                    leftBlocks[site + 1] = leftBlocks[site].nextBlock(psiGround,
                                           rightBlocks[lSys - 4 - site],
                                           false, false,
                                           rightBlocks[lSys - 5 - site]);
                reflectPredictedPsi(psiGround, mMax, rightBlocks[skips].m);
                               // reflect the system to reverse sweep direction
                for(int site = skips, end = lSys - 4 - skips; site < end; site++)
                    rightBlocks[site + 1] = rightBlocks[site].nextBlock(psiGround,
                                            leftBlocks[lSys - 4 - site],
                                            false, false,
                                            leftBlocks[lSys - 5 - site]);
                reflectPredictedPsi(psiGround, mMax, leftBlocks[skips].m);
                for(int site = skips, end = lSFinal - 1; site < end; site++)
                    leftBlocks[site + 1] = leftBlocks[site].nextBlock(psiGround,
                                           rightBlocks[lSys - 4 - site],
                                           false, false,
                                           rightBlocks[lSys - 5 - site]);
                std::cout << "Sweep " << i << " complete." << std::endl;
            };
        };
        EffectiveHamiltonian hSuperFinal
            = leftBlocks[lSFinal - 1].createHSuperFinal(rightBlocks[lEFinal - 1],
                                                        psiGround, lSFinal - 1,
                                                        skips);
                                               // calculate ground-state energy
        fileout << "Ground state energy density = "
                << hSuperFinal.gsEnergy / lSys << std::endl << std::endl;
        if(calcObservables)
        {
            std::cout << "Calculating observables..." << std::endl;
            VectorXd oneSiteVals;
            MatrixXd twoSiteVals;
            if(calcOneSiteExpValues)   // calculate one-site expectation values
                oneSiteVals = oneSiteExpValues(oneSiteOp, rangeOfObservables,
                                               lSys, hSuperFinal, leftBlocks,
                                               rightBlocks, fileout);
            if(calcTwoSiteExpValues)   // calculate two-site expectation values
                twoSiteVals = twoSiteExpValues(firstTwoSiteOp, secondTwoSiteOp,
                                               rangeOfObservables, lSys,
                                               hSuperFinal, leftBlocks,
                                               rightBlocks, fileout);
            if(calcOneSiteExpValues && calcTwoSiteExpValues)
            {
                MatrixXd connectedCorrFunc
                    = twoSiteVals - oneSiteVals * oneSiteVals.transpose();
                for(int i = 0, end = rangeOfObservables * rangeOfObservables;
                    i < end; i++)
                    if(std::abs(connectedCorrFunc(i)) < observableThreshold)
                        connectedCorrFunc(i) = 0.;
                fileout << "Connected correlation function:\n"
                        << connectedCorrFunc << std::endl << std::endl;
            };
        };
        std::cout << std::endl;
        clock_t stopTrial = clock();
        fileout << "Elapsed time: "
                << double(stopTrial - startTrial)/CLOCKS_PER_SEC << " s"
                << std::endl;
        fileout.close();
    };
    filein.close();
    
    clock_t stop = clock();
    std::cout << "Done. Elapsed time: " << double(stop - start)/CLOCKS_PER_SEC
              << " s" << std::endl;

    return 0;
}
