//
//   Example of a program to fit non-equidistant data points
//   =======================================================
//
//   The fitting function fcn is a simple chisquare function
//   The data consists of 5 data points (arrays x,y,z) + the errors in errorsz
//   More details on the various functions or parameters for these functions
//   can be obtained in an interactive ROOT session with:
//    Root > TMinuit *minuit = new TMinuit(10);
//    Root > minuit->mnhelp("*")  to see the list of possible keywords
//    Root > minuit->mnhelp("SET") explains most parameters
//

#include "TMinuit.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

using namespace std;

void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag);
Double_t func(Double_t y0, Double_t y1, Double_t *par);

Double_t x_data[10000], x_sample0[10000], x_sample1[10000];
Double_t y_data[10000], y_sample0[10000], y_sample1[10000];
Double_t x[10000], y[10000];

int ndata,nsample0,nsample1;


int nbins;
int iter;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    Int_t i;

    Double_t err = 1.00;

    iter++;
    if(iter%10==0) cerr << "iter: " << iter << "\r";

    //calculate chisquare
    Double_t chisq = 0;
    Double_t delta;
    for (i=0;i<ndata; i++) 
    {
        delta  = (y_data[i]-func(y_sample0[i],y_sample1[i],par))/err;
        //cerr << delta << endl;
        chisq += delta*delta;
    }
    cerr << chisq << endl;
    f = chisq;
}

//______________________________________________________________________________
Double_t func(Double_t y0, Double_t y1, Double_t *par)
{
    //Double_t value= pow(cos(par[0]),2)*y0 + pow(sin(par[0]),2)*y1;
    Double_t value= (1.0/(1.0+(par[0]*par[0]))) * ((par[0]*par[0])*y0 + y1);
    return value;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{

    ifstream INdata(argv[1]);
    ifstream INsample0(argv[2]);
    ifstream INsample1(argv[3]);

    Double_t junk;

    Double_t metric_size = 0.005;
    if (argc>4)
    {
        metric_size = atof(argv[4]);
    }

    // Set to 0's
    for (int i=0;i<10000;i++)
    {
        x_data[i]=0;
        x_sample0[i]=0;
        x_sample1[i]=0;

        y_data[i]=0;
        y_sample0[i]=0;
        y_sample1[i]=0;
    }

    // Read in the data
    ndata = 0;
    while(INdata >> x_data[ndata])
    {
        ndata++;
    }

    // Read in the data
    nsample0 = 0;
    while(INsample0 >> x_sample0[nsample0])
    {
        nsample0++;
    }

    // Read in the data
    nsample1 = 0;
    while(INsample1 >> x_sample1[nsample1])
    {
        nsample1++;
    }


    //////////////////////////////////////////////////////////////////////////////////////
    // Calculate the number of nearest neighbors;

    for(int i=0;i<ndata;i++)
    {
        for(int j=0;j<ndata;j++)
        {
            if (i!=j)
            {
                if (fabs(x_data[i]-x_data[j])<metric_size)
                {
                    y_data[i]++;
                }
            }
        }
    }
    // Normalize
    //cerr << "\nDATA ----------------------" << ndata << endl;
    for(int i=0;i<ndata;i++)
    {
        //cerr << y_data[i] << "\t";
        y_data[i] /= Double_t(ndata);
        //cerr << y_data[i] << "\t";
        //if (i%12==0) cerr << "\n";
    }

    ///////////////////////////////
    for(int i=0;i<ndata;i++)
    {
        for(int j=0;j<nsample0;j++)
        {
            if (1)
            {
                if (fabs(x_data[i]-x_sample0[j])<metric_size)
                {
                    y_sample0[i]++;
                }
            }
        }
    }
    // Normalize
    //cerr << "\nSAMPLE 0 ----------------------" << endl;
    for(int i=0;i<ndata;i++)
    {
        y_sample0[i] /= Double_t(nsample0);
        //cerr << y_sample0[i] << "\t";
        //if (i%12==0) cerr << "\n";
    }

    ///////////////////////////////

    for(int i=0;i<ndata;i++)
    {
        for(int j=0;j<nsample1;j++)
        {
            if (1)
            {
                if (fabs(x_data[i]-x_sample1[j])<metric_size)
                {
                    y_sample1[i]++;
                }
            }
        }
    }
    // Normalize
    //cerr << "\nSAMPLE 1 ----------------------" << endl;
    for(int i=0;i<ndata;i++)
    {
        y_sample1[i] /= Double_t(nsample1);
        //cerr << y_sample1[i] << "\t";
        //if (i%12==0) cerr << "\n";
    }

    ///////////////////////////////
    ///////////////////////////////////////////////////////////////////////////



    nbins = 10;

    iter = 0;

    TMinuit *gMinuit = new TMinuit(5);  //initialize TMinuit with a maximum of 5 params
    gMinuit->SetFCN(fcn);

    Double_t arglist[10];
    Int_t ierflg = 0;

    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

    // Set starting values and step sizes for parameters
    static Double_t vstart[3] = {0.90, 10.0 , 10.0 };
    static Double_t step[3] = {0.1 , 0.1 , 0.01 };
    gMinuit->mnparm(0, "a1", vstart[0], step[0], 0, 0, ierflg);
    //gMinuit->mnparm(1, "a2", vstart[1], step[1], 0,0,ierflg);
    //gMinuit->mnparm(2, "a3", vstart[2], step[2], 0,0,ierflg);

    // Now ready for minimization step
    arglist[0] = 500;
    arglist[1] = 1.;
    // Call the minimization
    gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);
    //gMinuit->mnmnos();
    //cerr << "---------------------- hi -------------------- " << endl;

    // Print results
    Double_t amin,edm,errdef;
    Int_t nvpar,nparx,icstat;
    gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
    gMinuit->mnprin(3,amin);

    Double_t parval,parerr;
    gMinuit->GetParameter(0,parval,parerr);
    //cerr << parval << " " << parerr << endl;
    cerr << "Fit values ----- " << endl;
    //cerr << cos(parval)*cos(parval) << " " << sin(parval)*sin(parval) << endl;
    cerr << parval*parval/(1.0 + parval*parval) << " " << 1.0/(1.0 + parval*parval) << endl;
    cerr << "Real values ----- " << endl;
    cerr << 1000.0/1100.0 << " " << 100.0/1100.0 << endl;

    return 0;
}



