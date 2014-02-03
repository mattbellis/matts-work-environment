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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag);
Double_t func(Double_t x,Double_t *par);
Double_t y[10000000],x[10000000];

int nbins;
int iter;

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  Int_t i;

  Double_t err = 1.0;

  iter++;
  if(iter%10==0) cerr << "iter: " << iter << "\r";

  //calculate chisquare
  Double_t chisq = 0;
  Double_t delta;
  for (i=0;i<nbins; i++) {
    delta  = (y[i]-func(x[i],par))/err;
    chisq += delta*delta;
  }
  f = chisq;
}

//______________________________________________________________________________
Double_t func(Double_t x,Double_t *par)
{
  Double_t value=(par[0]*log(x) + par[1]*pow(sin(x),2) + par[2]*pow(cos(x),2));
  return value;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{

  ifstream IN(argv[1]);

  Double_t junk;

  int i = 0;

  while(IN >> x[i])
  {
    IN >> y[i] >> junk;
    i++;
    if(i%10000==0) cerr << i << "\r";
  }
  nbins = i;

  iter = 0;

  TMinuit *gMinuit = new TMinuit(5);  //initialize TMinuit with a maximum of 5 params
  gMinuit->SetFCN(fcn);

  Double_t arglist[10];
  Int_t ierflg = 0;

  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

  // Set starting values and step sizes for parameters
  static Double_t vstart[3] = {10.0, 10.0 , 10.0 };
  static Double_t step[3] = {0.1 , 0.1 , 0.01 };
  gMinuit->mnparm(0, "a1", vstart[0], step[0], 0,0,ierflg);
  gMinuit->mnparm(1, "a2", vstart[1], step[1], 0,0,ierflg);
  gMinuit->mnparm(2, "a3", vstart[2], step[2], 0,0,ierflg);

  // Now ready for minimization step
  arglist[0] = 500;
  arglist[1] = 1.;
  // Call the minimization
  gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);

  // Print results
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
  gMinuit->mnprin(3,amin);

  return 0;
}



