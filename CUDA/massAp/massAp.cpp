#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "Riostream.h"
#include "TFile.h"
#include "TH2.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

/////////////////////////////////////////////////
// I want to calculate the aperture mass for a give shear field. 
// I have a catalogue of points and their gamma1/gamma2 shear values
// (this could correspond to a measurement, or the simulation I"m using here).
// For each point, the aperture mass is calculated by summing the conributions
// of all the surrounding points. In reality, the contribution tails off 
// after a few arcmin so you don't need to caluclate EVERY point's contrubution
// I'm going to send this calculation to the GPU
// The calculation is based upon eqns 3 and 9 of Dietrich+Hartlap(2009). 
//////////////////////////////////////////////////

double execute_kernel_gpu(matrix<float> this_gamma1, matrix<float> this_gamma2);

/*
 *  Main body of code
 */

int main(int argc, char **argv) { 

    char* filename;

    if (argc>1)
    {
        filename = argv[1];
    }

    else
    {
        printf("Usage: %s <filename >  \n",  argv[0]);
        return 1;
    }

    printf ("Parsed the command line\n");  

    ///////////////////////////////////////////////////////////////////////////
    /////now read in the points from the file.
    ///////////////////////////////////////////////////////////////////////////

    //make a root file to save all this 
    TFile *f = new TFile("massAp.root","recreate");
    TH2F *th2_kappa = new TH2F("th2_kappa", "kappa from mass ap", 1025, 0, 1024, 1025, 0, 1024);
    TH2F *th2_gamma1 = new TH2F("th2_gamma1", "gamma1", 1025, 0, 1024, 1025, 0, 1024);
    TH2F *th2_gamma2 = new TH2F("th2_gamma2", "gamma2", 1025, 0, 1024, 1025, 0, 1024);

    //I'm going to use matrices because they're convenient
    matrix<float> gamma1 (1024,1024);
    matrix<float> gamma2 (1024,1024);

    //read in the file
    ifstream infile;
    infile.open(filename);

    printf("%s\n", filename);

    int i = 0;
    float x=0, y=0, g1=0, g2=0;      
    int col=0, row=0;

    while(1)
    {
        if (i>=1024*1024) break;
        infile>>x>>y>>g1>>g2;

        if(col==1024){
            col=0;
            row++;
            gamma1(row, col) = g1;
            gamma2(row, col) = g2;
            th2_gamma1->Fill(row,col,g1);
            th2_gamma2->Fill(row,col,g2);
            col++;
        }else{
            gamma1(row, col) = g1;
            gamma2(row, col) = g2;
            th2_gamma1->Fill(row,col,g1);
            th2_gamma2->Fill(row,col,g2);
            col++;
        }

        i += 1;
        if(!infile.good()) break;	  
    }


    printf("number of entries: %d \n", i);


    ////////// now loop over each point in my 1024x1024 grid. 
    ////////// for each point, I want to make a sub-matrix 
    ///////// of the 64x64 grid around this point
    ////////// and pass that grid to the gpu to do the calculation
    ///////// there'll be 1024x1024 gpu calls this way 
    ///////// which is not th emost efficient way to do things! 

    //////// note that I want a range of 32 pix to do the summation over. 
    //////// 1pix = 10.5 arcsec, so that corresponds to 5.6arcmin
    //////// that's pretty much the limit of the range that could contribute
    //////// to the mass meaasurement at this point

    int sqsize = 32*2 + 1;
    matrix<float> this_gamma1 (sqsize,sqsize);
    matrix<float> this_gamma2 (sqsize,sqsize);
    double gpuOut = 0;
    matrix<float> kappa(1024-64, 1024-64);
    for(int i=32;i<1024-32;i++){
        printf("i: %d\n", i);
        for(int j=32;j<1024-32;j++){
            this_gamma1 = subrange(gamma1, i-32,i+32, j-32,j+32);
            this_gamma2 = subrange(gamma2, i-32,i+32, j-32,j+32);

            gpuOut = execute_kernel_gpu(this_gamma1, this_gamma2);
            kappa(i-32,j-32) = gpuOut;
            th2_kappa->Fill(i,j,gpuOut);
        }
    }

    th2_kappa->Write();
    th2_gamma1->Write();
    th2_gamma2->Write();
    f->Close();
}
