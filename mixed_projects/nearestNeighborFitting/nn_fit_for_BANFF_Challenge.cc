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
#include "TRandom3.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

using namespace std;

void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag);
Double_t func(Double_t y0, Double_t y1, Double_t y2, Double_t *par);
void create_subsamples(Int_t nsubsamples);
void calc_nn(Double_t* x0, Double_t* x1, Double_t* y, Double_t* yerr, Int_t nx0, Int_t nx1, Bool_t are_x0_and_x1_the_same, Double_t distance);
//void return_confidence_limits(Double_t* x, Int_t nx, Double_t* limits, Float_t confidence_interval);
void fill_confidence_limits(Float_t confidence_interval);

#define MC_SAMPLE_SIZES 5000
#define DATA_MAX_SIZE 2000

Double_t metric_size;

Double_t x_mc_sample[3][MC_SAMPLE_SIZES];
Double_t x_data[DATA_MAX_SIZE];

Double_t y_mc_sample[3][DATA_MAX_SIZE];
Double_t y_data[DATA_MAX_SIZE];

//Double_t x[MC_SAMPLE_SIZES], y[MC_SAMPLE_SIZES];

// For bootstrapping
Double_t x_mc_subsamples[3][1000][MC_SAMPLE_SIZES];
Double_t x_data_subsamples[1000][DATA_MAX_SIZE];

Double_t y_mc_subsamples[3][1000][DATA_MAX_SIZE];
Double_t y_data_subsamples[1000][DATA_MAX_SIZE];

Double_t mc_cl_intervals_lo[3][DATA_MAX_SIZE];
Double_t mc_cl_intervals_hi[3][DATA_MAX_SIZE];
Double_t data_cl_intervals_lo[DATA_MAX_SIZE];
Double_t data_cl_intervals_hi[DATA_MAX_SIZE];

Double_t mc_err[3][DATA_MAX_SIZE];
Double_t data_err[DATA_MAX_SIZE];

int ndata,nsample[3];

int iter;

////////////////////////////////////////////////////////////////////////////////
// Create the MC samples
////////////////////////////////////////////////////////////////////////////////
void create_subsamples(Int_t nsubsamples=1000)
{
    TRandom3 rnd;
    int index=0;

    for (int i=0;i<3;i++)
    {
        for (int j=0;j<nsubsamples;j++)
        {
            // Monte Carlo subsamples
            for (int k=0;k<MC_SAMPLE_SIZES;k++)
            {
                // Generate random number between 0 and MC_SAMPLE_SIZES-1
                index = (int)(MC_SAMPLE_SIZES*rnd.Rndm());
                x_mc_subsamples[i][j][k] = x_mc_sample[i][index];
            }

            // Data subsamples
            if (i==0) // We only have 1 data samples
            {
                for (int k=0;k<ndata;k++)
                {
                    // Generate random number between 0 and MC_SAMPLE_SIZES-1
                    index = (int)(ndata*rnd.Rndm());
                    x_data_subsamples[j][k] = x_data[index];
                }
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Compare routine needed by qsort.
////////////////////////////////////////////////////////////////////////////////
int compare (const void * a, const void * b)
{
    return (int)(1000000*( *(double*)a - *(double*)b ));
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// For bootstrapping, return the confidence interval.
////////////////////////////////////////////////////////////////////////////////
//void return_confidence_limits(Double_t* x, Int_t nx, Double_t* limits, Float_t confidence_interval=0.68)
//{
//qsort (x, nx, sizeof(Double_t), compare);
//}

////////////////////////////////////////////////////////////////////////////////
// For bootstrapping, return the confidence interval.
////////////////////////////////////////////////////////////////////////////////
void fill_confidence_limits(Float_t confidence_interval=0.68)
{
    int nsubsamples = 1000;
    double temp_array[1000];

    float cl_outer_band_size = (1.0 - confidence_interval)/2.0;
    int index_lo = (int)(nsubsamples*cl_outer_band_size);
    int index_hi = (int)(nsubsamples*(1.0-cl_outer_band_size));
    cerr << "index_lo/index_hi: " << cl_outer_band_size << " " << index_lo << " " << index_hi << endl;

    //////////////////////////////////////////////////////////////////////////////////////
    // Calculate the number of nearest neighbors;
    // in data.
    for (int j=0;j<nsubsamples;j++)
    {
        if (j%100==0) cerr << j << endl;
        //calc_nn(x_data,x_data_subsamples[j],y_data_subsamples[j],ndata,ndata,kTRUE,metric_size);
    }
    cerr << "Calced nn for data subsamples" << endl;
    //
    // in MC.
   
    for(int i=0;i<3;i++)
    {
        for (int j=0;j<nsubsamples;j++)
        {
            if (j%100==0) cerr << i << " " << j << endl;
            //calc_nn(x_data,x_mc_subsamples[i][j],y_mc_subsamples[i][j],ndata,nsample[i],kFALSE,metric_size);
        }
    }
    
    //////////////////////////////////////////////////////////////

    for (int i=0;i<3;i++)
    {
        // Monte Carlo subsamples
        for (int k=0;k<ndata;k++)
        {
            for (int j=0;j<nsubsamples;j++)
            {
                // Generate random number between 0 and MC_SAMPLE_SIZES-1
                temp_array[j] = y_mc_subsamples[i][j][k];
            }

            qsort (temp_array, nsubsamples, sizeof(Double_t), compare);
            mc_cl_intervals_lo[i][k] = temp_array[index_lo];
            mc_cl_intervals_hi[i][k] = temp_array[index_hi];
            mc_err[i][k] = (mc_cl_intervals_hi[i][k] - mc_cl_intervals_lo[i][k])/2.0;
            cerr << "mc_err " << i << " " << k << " " << mc_err[i][k] << endl;


            // Data subsamples
            if (i==0) // We only have 1 data samples
            {
                for (int j=0;j<nsubsamples;j++)
                {
                    // Generate random number between 0 and MC_SAMPLE_SIZES-1
                    temp_array[j] = y_data_subsamples[j][k];
                }

                qsort (temp_array, nsubsamples, sizeof(Double_t), compare);
                //for (int j=0;j<nsubsamples;j++)
                    //cerr << temp_array[j] << " ";
                //cerr << "\n";
                data_cl_intervals_lo[k] = temp_array[index_lo];
                data_cl_intervals_hi[k] = temp_array[index_hi];
                data_err[k] = (data_cl_intervals_hi[k] - data_cl_intervals_lo[k])/2.0;
                cerr << "data_err " << k << " " << x_data[k] << " " << y_data[k] << " " << data_err[k] << endl;

            }
        }
    }   
    //qsort (x, nx, sizeof(Double_t), compare);
}

////////////////////////////////////////////////////////////////////////////////
// Calculate the number of nearest neighbors;
////////////////////////////////////////////////////////////////////////////////
void calc_nn(Double_t* x0, Double_t* x1, Double_t* y0, Double_t* y0_err, Int_t nx0, Int_t nx1, Bool_t are_x0_and_x1_the_same=kFALSE, Double_t distance=0.05)
{
    ////////////////////////////////////////////////////////////////////////////
    // Calc the number of nearest neighbors.
    ////////////////////////////////////////////////////////////////////////////
    for(int i=0;i<nx0;i++)
    {
        for(int j=0;j<nx1;j++)
        {
            if (are_x0_and_x1_the_same==kTRUE && i==j)
            {
                // Don't do anything.
                // This is to make sure that we don't count the self-distance
                // if we're working with the same dataset.
            }
            else
            {
                // Try adaptive binning
                /*
                if (x0[i]<0.02)
                {
                    distance = 0.001;
                }
                else if (x0[i]>=0.02 && x0[i]<0.10)
                {
                    distance = 0.02;
                }
                else if (x0[i]>=0.10)
                {
                    distance = 0.05;
                }
                */

                if (fabs(x0[i]-x1[j])<distance)
                {
                    y0[i]++;
                }
            }
        }
        y0_err[i] = sqrt(y0[i])/1.2;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Normalize to the number of events in the x1 sample.
    // This will tell us what percentage of the x1 events, are within
    // the metric size of a given x0 data point.
    ////////////////////////////////////////////////////////////////////////////
    Double_t norm_factor = (Double_t)nx1;
    for(int i=0;i<nx0;i++)
    {
        //cerr << y0[i] << endl;
        y0[i] /= norm_factor;
        y0_err[i] /= norm_factor;
        //cerr << y0[i] << endl;
    }

}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    Int_t i;

    Double_t err = 0.01;

    iter++;
    //if(iter%10==0) cerr << "iter: " << iter << "\r";

    //calculate chisquare
    Double_t chisq = 0;
    Double_t delta;
    for (i=0;i<ndata; i++) 
    {
        //data_err[i] = sqrt(y_data[i]);
        //mc_err[0][i] = sqrt(y_mc_sample[0][i]);
        //mc_err[1][i] = sqrt(y_mc_sample[1][i]);
        //mc_err[2][i] = sqrt(y_mc_sample[2][i]);

        err = data_err[i]*data_err[i] + mc_err[0][i]*mc_err[0][i] + mc_err[1][i]*mc_err[1][i] + mc_err[2][i]*mc_err[2][i];
        //err = data_err[i]*data_err[i] ;
        //cerr << "err: " << err << endl;

        delta  = (y_data[i]-func(y_mc_sample[0][i],y_mc_sample[1][i],y_mc_sample[2][i],par));
        //cerr << delta << endl;

        chisq += delta*delta/err;
    }
    //cerr << chisq << endl;
    f = chisq;
    //f = chisq + 1.0 - par[0] - par[1] - par[2];
}

//______________________________________________________________________________
Double_t func(Double_t y0, Double_t y1, Double_t y2, Double_t *par)
{
    //Double_t value= pow(cos(par[0]),2)*y0 + pow(sin(par[0]),2)*y1;
    //Double_t a0_sq = par[0]*par[0];
    //Double_t a1_sq = par[1]*par[1];
    //Double_t value= (1.0/(1.0+a0_sq+a1_sq)) * (a0_sq*y0 + a1_sq*y1 + y2);

    //Double_t value = (1.0/(par[0]+par[1]))*(par[0]*y0 + par[1]*y1 + par[2]*y2);
    //Double_t value = (par[0]*y0 + par[1]*y1 + (1.0-par[0]-par[1])*y2);
    Double_t value = (par[0]*y0 + par[1]*y1 + par[2]*y2);
    value += 1.0 - par[0] - par[1] - par[2];

    return value;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{

    ////////////////////////////////////////////////////////////////////////////
    // Read in the command line arguments.
    ////////////////////////////////////////////////////////////////////////////
    ifstream INdata(argv[1]);

    ifstream INsample[3];
    INsample[0].open(argv[2]);
    INsample[1].open(argv[3]);
    INsample[2].open(argv[4]);

    Double_t junk;

    //Double_t metric_size = 0.005;
    if (argc>4)
    {
        metric_size = atof(argv[5]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Zero out our matrices.
    ////////////////////////////////////////////////////////////////////////////
    for (int i=0;i<10000;i++)
    {
        x_data[i]=0;
        y_data[i]=0;

        for (int j=0;j<3;j++)
        {
            x_mc_sample[j][i]=0;
            y_mc_sample[j][i]=0;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Read in the data
    ////////////////////////////////////////////////////////////////////////////
    ndata = 0;
    while(INdata >> x_data[ndata])
    {
        ndata++;
    }

    // Read in the sample distributions.
    for (int j=0;j<3;j++)
    {
        nsample[j] = 0;
        while(INsample[j] >> x_mc_sample[j][nsample[j]])
        {
            nsample[j]++;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Calculate the number of nearest neighbors;
    // in data.
    calc_nn(x_data,x_data,y_data,data_err,ndata,ndata,kTRUE,metric_size);
    //
    // in MC.
    for(int j=0;j<3;j++)
    {
        calc_nn(x_data,x_mc_sample[j],y_mc_sample[j],mc_err[j],ndata,nsample[j],kFALSE,metric_size);
    }
    //////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////
    // Create MC sub samples for bootstrapping
    ///////////////////////////////////////////////////////////////////////////
    cerr << "Creating subsamples for bootstrapping......." << endl;
    //create_subsamples(1000);
    cerr << "Created subsamples for bootstrapping......." << endl;
    ///////////////////////////////////////////////////////////////////////////
    //Double_t data_cl_intervals_lo[1000][DATA_MAX_SIZE];
    //Double_t data_cl_intervals_hi[1000][DATA_MAX_SIZE];
    //return_confidence_limits(y_data_subsamples,data_cl_intervals_lo,data_,ndata);
    //fill_confidence_limits(0.68);
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // MINIMIZE!!!!!!!!!!!!!!!!!!!!!! ------- !!!!!!!!!!!!!!!!!!!
    ///////////////////////////////////////////////////////////////////////////
    iter = 0;

    TMinuit *gMinuit = new TMinuit(5);  //initialize TMinuit with a maximum of 5 params
    gMinuit->SetFCN(fcn);

    Double_t arglist[10];
    Int_t ierflg = 0;

    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

    arglist[0] = 2;
    gMinuit->mnexcm("SET STR", arglist ,1,ierflg);

    // Set starting values and step sizes for parameters
    static Double_t vstart[3] = {0.90, 0.02 , 0.2 };
    static Double_t step[3] = {0.01 , 0.01 , 0.01 };

    //gMinuit->mnparm(0, "a1", vstart[0], step[0], 0, 0, ierflg);
    //gMinuit->mnparm(1, "a2", vstart[1], step[1], 0, 0, ierflg);

    gMinuit->mnparm(0, "a1", vstart[0], step[0], 0.0, 1.0, ierflg);
    gMinuit->mnparm(1, "a2", vstart[1], step[1], 0.0, 1.0, ierflg);
    gMinuit->mnparm(2, "a3", vstart[2], step[2], 0.0, 1.0, ierflg);


    // Now ready for minimization step
    arglist[0] = 10000;
    arglist[1] = 1.;
    // Call the minimization
    gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);
    gMinuit->mnexcm("MINOS", arglist ,2,ierflg);
    //gMinuit->mnmnos();
    //cerr << "---------------------- hi -------------------- " << endl;

    // Print results
    Double_t amin,edm,errdef;
    Int_t nvpar,nparx,icstat;
    gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
    gMinuit->mnprin(3,amin);

    Double_t parval[3],parerr[3];
    gMinuit->GetParameter(0,parval[0],parerr[0]);
    gMinuit->GetParameter(1,parval[1],parerr[1]);
    gMinuit->GetParameter(2,parval[2],parerr[2]);
    //cerr << parval << " " << parerr << endl;
    cerr << "Fit values ----- " << endl;
    //Double_t a0_sq = parval[0]*parval[0];
    //Double_t a1_sq = parval[1]*parval[1];
    //Double_t norm = (1.0+a0_sq+a1_sq);
    //cerr << cos(parval)*cos(parval) << " " << sin(parval)*sin(parval) << endl;

    Double_t norm = parval[0] + parval[1] + parval[2];
    cerr << "ndata: " << ndata << endl;
    for (int i=0;i<3;i++)
    {
        cerr << "par: " << i << " " << parval[i] << " " << parerr[i] << endl;
    }
    cerr << "norm: " << norm << endl;
    cerr << "a0: " << parval[0]/norm << endl;
    cerr << "a1: " << parval[1]/norm << endl;
    cerr << "a2: " << parval[2]/norm << endl;

    return 0;
}



