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
void create_bootstrap_samples(Int_t nbootstrap_samples);
void calc_nn(Double_t* x0, Double_t* x1, Double_t* y, Double_t* yerr, Int_t nx0, Int_t nx1, Bool_t are_x0_and_x1_the_same,\
        Bool_t use_self_metric, Double_t distance);
//void return_confidence_limits(Double_t* x, Int_t nx, Double_t* limits, Float_t confidence_interval);
void calculate_nn_for_bootstrap_samples();

#define MC_SAMPLE_SIZES 10000
#define DATA_MAX_SIZE 2000
#define NUM_BOOTSTRAP_SAMPLES 50

Int_t nfits = 0;

Double_t metric_size;

Double_t mc_sample[3][MC_SAMPLE_SIZES];
Double_t data[DATA_MAX_SIZE];

Double_t pdf_mc_sample[3][DATA_MAX_SIZE];
Double_t pdf_data[DATA_MAX_SIZE];

//Double_t x[MC_SAMPLE_SIZES], y[MC_SAMPLE_SIZES];

// For bootstrapping
Double_t mc_bootstrap_samples[3][NUM_BOOTSTRAP_SAMPLES][MC_SAMPLE_SIZES];
Double_t data_bootstrap_samples[NUM_BOOTSTRAP_SAMPLES][DATA_MAX_SIZE];

Double_t pdf_mc_bootstrap_samples[3][NUM_BOOTSTRAP_SAMPLES][DATA_MAX_SIZE];
Double_t pdf_data_bootstrap_samples[NUM_BOOTSTRAP_SAMPLES][DATA_MAX_SIZE];

Double_t mc_cl_intervals_lo[3][DATA_MAX_SIZE];
Double_t mc_cl_intervals_hi[3][DATA_MAX_SIZE];
Double_t data_cl_intervals_lo[DATA_MAX_SIZE];
Double_t data_cl_intervals_hi[DATA_MAX_SIZE];

Double_t mc_err[3][DATA_MAX_SIZE];
Double_t data_distances[DATA_MAX_SIZE];

Double_t mc_bootstrap_samples_integral[3][NUM_BOOTSTRAP_SAMPLES];

int ndata,nsample[3];

int iter;

////////////////////////////////////////////////////////////////////////////////
// Create the MC samples
////////////////////////////////////////////////////////////////////////////////
void create_bootstrap_samples(Int_t nbootstrap_samples=NUM_BOOTSTRAP_SAMPLES)
{
    TRandom3 rnd;
    int index=0;

    for (int i=0;i<3;i++)
    {
        for (int j=0;j<nbootstrap_samples;j++)
        {
            // Monte Carlo bootstrap_samples
            for (int k=0;k<nsample[i];k++)
            {
                // Generate random number between 0 and MC_SAMPLE_SIZES-1
                index = (int)(nsample[i]*rnd.Rndm());
                mc_bootstrap_samples[i][j][k] = mc_sample[i][index];
            }

            // Data bootstrap_samples
            if (i==0) // We only have 1 data samples
            {
                for (int k=0;k<ndata;k++)
                {
                    // Generate random number between 0 and MC_SAMPLE_SIZES-1
                    index = (int)(ndata*rnd.Rndm());
                    data_bootstrap_samples[j][k] = data[index];
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
void calculate_nn_for_bootstrap_samples()
{
    int nbootstrap_samples = NUM_BOOTSTRAP_SAMPLES;
    double temp_array[NUM_BOOTSTRAP_SAMPLES];

    //////////////////////////////////////////////////////////////////////////////////////
    // in MC.
    for(int i=0;i<3;i++)
    {
        for (int j=0;j<nbootstrap_samples;j++)
        {
            if (j%1000==0) cerr << i << " " << j << endl;

            calc_nn(data,mc_bootstrap_samples[i][j],pdf_mc_bootstrap_samples[i][j],mc_err[i],ndata,nsample[i],kFALSE,kFALSE,metric_size);

            // Calculate integral
            Double_t integral = 0.0;
            for(int k=0;k<ndata;k++)
            {
                //if (i==0 && j==0)
                //{
                //cerr << "data_distances: " << data[k] << " " << data_distances[k] << endl;
                //}

                integral += pdf_mc_bootstrap_samples[i][j][k]*data_distances[k];    
                //integral += pdf_mc_bootstrap_samples[i][j][k];    
            }
            //cerr << i << " " << j << " integral: " << integral << endl;
            mc_bootstrap_samples_integral[i][j] = integral;

        }
    }

    //////////////////////////////////////////////////////////////

}

////////////////////////////////////////////////////////////////////////////////
// Calculate the number of nearest neighbors;
////////////////////////////////////////////////////////////////////////////////
void calc_nn(Double_t* x0, Double_t* x1, Double_t* y0, Double_t* y0_err, Int_t nx0, Int_t nx1, Bool_t are_x0_and_x1_the_same=kFALSE, \
        Bool_t use_self_metric=kFALSE, Double_t distance=0.05)
{
    ////////////////////////////////////////////////////////////////////////////
    // Calc the number of nearest neighbors.
    ////////////////////////////////////////////////////////////////////////////
    Double_t self_metric[nx0];
    Double_t temp_array[nx0];
    Double_t tempx = 0.0;
    Double_t neighbor_lo = 0.0;
    Double_t neighbor_hi = 0.0;


    ///*
    if (use_self_metric)
    {
        for (int i=0;i<nx0;i++)
        {
            temp_array[i] = x0[i];
        }
        qsort (temp_array, nx0, sizeof(Double_t), compare);
        for (int i=0;i<nx0;i++)
        {
            tempx = x0[i];
            for (int j=0;j<nx0;j++)
            {
                if (tempx==temp_array[j])
                {
                    if (j==0)
                    {
                        // I can do this because I know the low edge is 0.0.
                        neighbor_lo = 0.0;
                        neighbor_hi = temp_array[j+1];
                    }
                    else if (j==nx0-1)
                    {
                        // I can do this because I know the high edge is 1.0.
                        neighbor_lo = temp_array[j-1];
                        neighbor_hi = 1.0;
                    }
                    else
                    {
                        neighbor_lo = temp_array[j-1];
                        neighbor_hi = temp_array[j+1];
                    }
                    break;
                }
            }
            self_metric[i] = (neighbor_hi-x0[i])/2.0 + (x0[i]-neighbor_lo)/2.0;
        }
    }
    //*/

    for(int i=0;i<nx0;i++)
    {
        for(int j=0;j<nx1;j++)
        {
            //cerr << "ij: " << i << " " << j << endl;
            if (are_x0_and_x1_the_same==kTRUE && i==j)
            {
                // Don't do anything.
                // This is to make sure that we don't count the self-distance
                // if we're working with the same dataset.
            }
            else
            {
                if (fabs(x0[i]-x1[j])<distance)
                {
                    y0[i]++;
                }
            }
        }
        //y0_err[i] = sqrt(y0[i])/1.2;
        if(use_self_metric)
        {
            y0_err[i] = self_metric[i];
        }
        else
        {
            y0_err[i] = 1.0;
        }
        //cerr << "data_and_err: " << i << " " << y0[i] << " " << y0_err[i] << endl;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Normalize to the number of events in the x1 sample.
    // This will tell us what percentage of the x1 events, are within
    // the metric size of a given x0 data point.
    ////////////////////////////////////////////////////////////////////////////
    //Double_t norm_factor = (Double_t)nx1;
    // Trying Doug's normalization.
    Double_t norm_factor = (Double_t)nx1*distance;
    //cerr << "norm_factor: " << norm_factor << endl;
    for(int i=0;i<nx0;i++)
    {
        //cerr << y0[i] << endl;
        y0[i] /= norm_factor;
        //y0_err[i] *= 1e7;
        //cerr << y0[i] << endl;
    }

}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    Int_t i;

    Double_t lh = 0.0;
    Double_t norm = 0.0;
    Double_t value = 0.0;
    Double_t y0, y1, y2;

    iter++;
    //if(iter%10==0) cerr << "iter: " << iter << "\r";

    // Calculate likelihood
    /*
    Double_t integral = 0.0;
    for (i=0;i<ndata; i++) 
    {
        integral += func(pdf_mc_sample[0][i],pdf_mc_sample[1][i],pdf_mc_sample[2][i],par)*data_distances[i];
    }
    */


    ////////////////////////////////////////////////////////////////////
    // This was working
    ////////////////////////////////////////////////////////////////////
    // Do it for the data
    //for (i=0;i<ndata; i++) 
    //{
        //value  = func(pdf_mc_sample[0][i],pdf_mc_sample[1][i],pdf_mc_sample[2][i],par);
        //lh  -= log(value/(1.0));
    //}

    // Do it for the bootstrapped samples
    for (i=0;i<ndata; i++) 
    {
        y0 = 0.0;
        y1 = 0.0;
        y2 = 0.0;
        for (int j=0;j<NUM_BOOTSTRAP_SAMPLES;j++)
        {
            y0 += pdf_mc_bootstrap_samples[0][j][i] / mc_bootstrap_samples_integral[0][j];
            y1 += pdf_mc_bootstrap_samples[1][j][i] / mc_bootstrap_samples_integral[1][j];
            y2 += pdf_mc_bootstrap_samples[2][j][i] / mc_bootstrap_samples_integral[2][j];

            //y0 += (pdf_mc_bootstrap_samples[0][j][i] / mc_bootstrap_samples_integral[0][j])*data_distances[i];
            //y1 += (pdf_mc_bootstrap_samples[1][j][i] / mc_bootstrap_samples_integral[1][j])*data_distances[i];
            //y2 += (pdf_mc_bootstrap_samples[2][j][i] / mc_bootstrap_samples_integral[2][j])*data_distances[i];

            //y0 += pdf_mc_bootstrap_samples[0][j][i];
            //y1 += pdf_mc_bootstrap_samples[1][j][i];
            //y2 += pdf_mc_bootstrap_samples[2][j][i];

            //value  = func(y0,y1,y2,par);
            //lh  -= log(value);
        }
        //cerr << "y012: " << y0 << " " << y1 << " " << y2 << endl;
        value  = func(y0,y1,y2,par);
        lh  -= log(value);
    }

    f = lh;
    //cerr << "lh: " << lh << endl;

    //f += 1000.0*(1.0 - par[0] - par[1] - par[2])*(1.0 - par[0] - par[1] - par[2]);

    // Penalty function to keep things normalized.
    if (nfits==0)
    {
        if (par[1] > 1.0 - par[0])
        {
            f = 1e8;
        }
    }
    else if (nfits==1)
    {
        if (par[0] > 0.99)
        {
            f = 1e8;
        }
    }

}

//______________________________________________________________________________
Double_t func(Double_t y0, Double_t y1, Double_t y2, Double_t *par)
{
    Double_t k1 = par[0];
    Double_t k2 = par[1];
    Double_t k3 = 1.0 - k1 - k2;
    //Double_t value = (par[0]*y0 + par[1]*y1 + par[2]*y2);

    if (nfits==1)
    {
        k2 = 1.0 - k1;
        k3 = 0.0;
    }

    Double_t value = (k1*y0 + k2*y1 + k3*y2);

    /*
    cerr << "value: " <<  value;
    cerr << " y0: " << y0;
    cerr << " y1: " << y1;
    cerr << " y2: " << y2;
    cerr << " k1: " << k1;
    cerr << " k2: " << k2;
    cerr << " k3: " << k3 << endl;
    */

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

    metric_size = 0.010;
    if (argc>4)
    {
        metric_size = atof(argv[5]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Zero out our matrices.
    ////////////////////////////////////////////////////////////////////////////
    for (int i=0;i<DATA_MAX_SIZE;i++)
    {
        data[i]=0;
        pdf_data[i]=0;

        for (int j=0;j<3;j++)
        {
            mc_sample[j][i]=0;
            pdf_mc_sample[j][i]=0;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Read in the data
    ////////////////////////////////////////////////////////////////////////////
    ndata = 0;
    while(INdata >> data[ndata])
    {
        ndata++;
    }

    // Read in the sample distributions.
    for (int j=0;j<3;j++)
    {
        nsample[j] = 0;
        while(INsample[j] >> mc_sample[j][nsample[j]])
        {
            nsample[j]++;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Calculate the number of nearest neighbors;
    // in data.
    calc_nn(data,data,pdf_data,data_distances,ndata,ndata,kTRUE,kTRUE,metric_size);
    //
    // in MC.
    for(int j=0;j<3;j++)
    {
        calc_nn(data,mc_sample[j],pdf_mc_sample[j],mc_err[j],ndata,nsample[j],kFALSE,kFALSE,metric_size);
    }
    //////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////
    // Create MC sub samples for bootstrapping
    ///////////////////////////////////////////////////////////////////////////
    cerr << "Creating bootstrap_samples for bootstrapping......." << endl;
    create_bootstrap_samples(NUM_BOOTSTRAP_SAMPLES);
    cerr << "Created bootstrap_samples for bootstrapping......." << endl;
    ///////////////////////////////////////////////////////////////////////////
    // Calculating all the probabilities for the bootstrap samples
    cerr << "Calculating probabilities for bootstrapped samples......." << endl;
    calculate_nn_for_bootstrap_samples();
    cerr << "Calculated  probabilities for bootstrapped samples......." << endl;
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // MINIMIZE!!!!!!!!!!!!!!!!!!!!!! ------- !!!!!!!!!!!!!!!!!!!
    ///////////////////////////////////////////////////////////////////////////
    iter = 0;

    Double_t nll[2];
    Double_t kvals[2][3];
    Double_t kvalerrs[2][3];
    Double_t nkvals[2][3];
    Double_t nkvalerrs[2][3];

    for (nfits=0;nfits<2;nfits++)
    {
        TMinuit *gMinuit = new TMinuit(5);  //initialize TMinuit with a maximum of 5 params
        gMinuit->SetFCN(fcn);

        Double_t arglist[10];
        Int_t ierflg = 0;

        // Set to be 0.5 for maximum likelihood.
        arglist[0] = 0.5;
        gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

        arglist[0] = 2;
        gMinuit->mnexcm("SET STR", arglist ,1,ierflg);

        // Set starting values and step sizes for parameters
        static Double_t vstart[3] = {0.90, 0.02 , 0.02 };
        static Double_t step[3] = {0.01 , 0.01 , 0.01 };

        //gMinuit->mnparm(0, "a1", vstart[0], step[0], 0, 0, ierflg);
        //gMinuit->mnparm(1, "a2", vstart[1], step[1], 0, 0, ierflg);

        if (nfits==0)
        {
            gMinuit->mnparm(0, "a1", vstart[0], step[0], 0.0, 1.0, ierflg);
            gMinuit->mnparm(1, "a2", vstart[1], step[1], 0.0, 1.0, ierflg);
        }

        else if (nfits==1)
        {
            gMinuit->mnparm(0, "a1", vstart[0], step[0], 0.0, 1.0, ierflg);
            gMinuit->mnparm(1, "a2", 0.0, 0.1, 0.0, 1.0, ierflg);
            gMinuit->mnfixp(1, ierflg);
        }

        //gMinuit->mnparm(2, "a3", vstart[2], step[2], 0.0, 1.0, ierflg);


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
        //gMinuit->GetParameter(2,parval[2],parerr[2]);
        //cerr << parval << " " << parerr << endl;
        cerr << "Fit values ----- " << endl;
        //Double_t a0_sq = parval[0]*parval[0];
        //Double_t a1_sq = parval[1]*parval[1];
        //Double_t norm = (1.0+a0_sq+a1_sq);
        //cerr << cos(parval)*cos(parval) << " " << sin(parval)*sin(parval) << endl;

        Double_t k1 = parval[0];
        Double_t k2 = parval[1];
        Double_t k3 = 1.0 - k1 - k2;
        if (nfits==1)
        {
            k2 = 1.0 - k1;
            k3 = 0.0;
        }

        Double_t norm = k1 + k2 + k3;
        cerr << "ndata: " << ndata << endl;
        for (int i=0;i<2;i++)
        {
            cerr << "par: " << i << " " << parval[i] << " " << parerr[i] << endl;
        }
        cerr << "norm: " << norm << endl;
        cerr << "k1: " << k1/norm << endl;
        cerr << "k2: " << k2/norm << endl;
        cerr << "k3: " << k3/norm << endl;

        cerr << "# k1: " << ndata*k1/norm << " +/- " << ndata*parerr[0]/norm << endl;
        cerr << "# k2: " << ndata*k2/norm << " +/- " << ndata*parerr[1]/norm << endl;
        cerr << "# k3: " << ndata*k3/norm << endl;

        kvals[nfits][0] = k1/norm;
        kvals[nfits][1] = k3/norm;
        kvals[nfits][2] = k2/norm;

        kvalerrs[nfits][0] = parerr[0];
        kvalerrs[nfits][1] = parerr[1];
        kvalerrs[nfits][2] = 0.0;

        nkvals[nfits][0] = ndata*k1/norm;
        nkvals[nfits][1] = ndata*k2/norm;
        nkvals[nfits][2] = ndata*k3/norm;

        nkvalerrs[nfits][0] = ndata*parerr[0]/norm;
        nkvalerrs[nfits][1] = ndata*parerr[1]/norm;
        nkvalerrs[nfits][2] = 0.0;

        nll[nfits] = amin;

    }

    cerr << " ------------------------- " << endl;
    cerr << " ------------------------- " << endl;
    for (int i=0;i<2;i++)
    {
    cerr << " ------------------------- " << endl;
    cerr << "nfit: " << i << "\tnll: " << nll[i] << endl;
    cerr << " ------------------------- " << endl;
    for (int j=0;j<3;j++)
    {
        cerr << "nfit: " << i << "\t# k" << j+1 << ": " << nkvals[i][j] << " +/- " << nkvalerrs[i][j] << endl;
    }
    }
    cerr << " ------------------------- " << endl;
    cerr << " ------------------------- " << endl;
    cerr << "significance: " << sqrt(2.0*(nll[1] - nll[0])) << endl;


    return 0;
}



