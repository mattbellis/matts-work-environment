#include <thrust/device_ptr.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <ctime>
#include <cmath>
#include <iostream>
#include <iterator>
#include <cstdlib>
#include "TRandom3.h"
#include "TMinuit.h" 

// global variables suck
const size_t N = 1<<17; // 31<<20; //=1048576
const size_t N_norm = 1<<17; // 31<<20; //=1048576

const float TWO_PI = 6.2831853071795862;

thrust::host_vector<float> vars(N) ;
thrust::host_vector<float> norm_vars(N_norm) ;
TMinuit* minuit = 0;
TRandom3 r;
thrust::device_vector<float> d_vars;// = vars;
thrust::device_vector<float> d_norm_vars;// = vars;

///////////////////////////////////////////////////////////////////////////////
// Make Random number between -5 and 5 
///////////////////////////////////////////////////////////////////////////////
float make_random_float(void)
{
    return (10.0*r.Rndm()) - 5.0;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Make Random number drawn from a Gaussian
///////////////////////////////////////////////////////////////////////////////
float make_random_float_gaus(void)
{
    return r.Gaus(-0.5,0.4);
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// We'll use a 5-tuple to store our data and variables
//typede thrust::tuple<float,float,float,float,float> Float5;
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// This is "the kernel"
///////////////////////////////////////////////////////////////////////////////
struct Fcn {
    template <typename Tuple>
        __host__ __device__
        void operator()(Tuple a)
        {
            // dummy Likelihood
            float x = thrust::get<1>(a);  // data
            float sigma = thrust::get<2>(a);
            float cent = thrust::get<3>(a);
            float norm = thrust::get<4>(a);

            //thrust::get<0>(a) = log((1./(sqrt(TWO_PI)*sigma))*norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));
            thrust::get<0>(a) = log(norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));

        }
};
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// This is "the norm kernel"
///////////////////////////////////////////////////////////////////////////////
struct Fcn_norm {
    template <typename Tuple>
        __host__ __device__
        void operator()(Tuple a)
        {
            // dummy Likelihood
            float x = thrust::get<1>(a);  // data
            float sigma = thrust::get<2>(a);
            float cent = thrust::get<3>(a);
            float norm = thrust::get<4>(a);

            //thrust::get<0>(a) = log((1./(sqrt(TWO_PI)*sigma))*norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));
            thrust::get<0>(a) = norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma));

        }
};
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void gpufcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    //f = ParFunc(par[0],par[1],par[2],5);
    // for this call to the function
    float norm = par[2];
    float sigmaval = par[1];
    float centval = par[0];

    //thrust::device_vector<float> d_vars= vars;

    // set all of these vectors to this.
    thrust::device_vector<float> d_sigma(N, sigmaval);
    thrust::device_vector<float> d_cent(N, centval);
    thrust::device_vector<float> d_norm(N, norm);

    // Another way to do this:
    //  thrust::device_vector<float> d_sigma(N);
    //  thrust::fill(d_sigma.begin(), d_sigma.end(), sigmaval);

    //result vector
    thrust::device_vector<float> result(N);

    // do it.
    thrust::for_each(thrust::make_zip_iterator(make_tuple(result.begin(),
                    d_vars.begin(), 
                    d_sigma.begin(), 
                    d_cent.begin(),
                    d_norm.begin())),
            thrust::make_zip_iterator(make_tuple(result.end(),
                    d_vars.end(), 
                    d_sigma.end(), 
                    d_cent.end(),
                    d_norm.end())),
            Fcn() );	
    // sum it up
    float sum = thrust::reduce(result.begin(), result.end());

    //std::cerr << sum << "\t" << thrust::reduce(d_sigma.begin(), d_sigma.end()) << std::endl;
    
    // Calc the normalization integral
    thrust::for_each(thrust::make_zip_iterator(make_tuple(result.begin(),
                    d_norm_vars.begin(), 
                    d_sigma.begin(), 
                    d_cent.begin(),
                    d_norm.begin())),
            thrust::make_zip_iterator(make_tuple(result.end(),
                    d_norm_vars.end(), 
                    d_sigma.end(), 
                    d_cent.end(),
                    d_norm.end())),
            Fcn_norm() );	

    // sum it up
    float norm_sum = thrust::reduce(result.begin(), result.end());

    //std::cerr << norm_sum << "\t" << thrust::reduce(d_sigma.begin(), d_sigma.end()) << std::endl;

    //f= (double) 2.*(-1.*sum + norm);
    float norm_integral =  1.0*((float)N/(float)N_norm)*(norm_sum);
    f = (float) (-1.*(sum - norm_integral));
    //std::cerr << "norm_sum: " << norm_sum << std::endl;
    std::cout<<"FCN="<<f<<":  norm="<<norm<<";  sigma = "<<sigmaval<<";  cent="<<centval << "; norm_integral: " << norm_integral <<std::endl;
} 
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main(void)
{
    // number of datapoints
    //const size_t N = 1<<24; // 31<<20; //=1048576

    std::cout<<"N objects:"<<N<<std::endl;
    std::cout<<"N_norm objects:"<<N_norm<<std::endl;

     // raw pointer to device memory
    int * raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(int));
    std::cerr<<"Allocated raw_ptr:"<<std::endl;



    // generate the mock data.  I think this is actually on the CPU

    ///////////////////////////////////////////////////////////////////////////
    // Generate the random numbers
    ///////////////////////////////////////////////////////////////////////////
    r.SetSeed(2112);
    thrust::generate(vars.begin(), vars.end(), make_random_float_gaus);  
    std::cerr << "Generated the data events...." << std::endl;
    d_vars = vars;
    std::cerr << "Copied over to the device...." << std::endl;
    // Generate MC for the normalization integral
    thrust::generate(norm_vars.begin(), norm_vars.end(), make_random_float);  
    std::cerr << "Generated the normalization events...." << std::endl;
    d_norm_vars = norm_vars;
    std::cerr << "Copied over to the device...." << std::endl;
    //for(int i = 0; i< N; i++){
    	//std::cout<<vars[i] << " " << norm_vars[i] <<std::endl;
    //}

    // move it to the GPU (now in the fcn)

    clock_t start = clock() ;

    // Define the minuit object.
    minuit = new TMinuit(3);
    minuit->SetFCN(gpufcn);

    float mean=-0.5;
    float sig = 0.2;
    float norm = 1.0;
    //float norm = 100;
    double vstrt[3]={mean,sig,norm}; // initial guesses for parameters
    double stp[3]={0.01,0.01,0.01}; // step size in the minimization
    double bmin[3]={0.0,0.0,0.0}; // minimum boundary
    double bmax[3]={0.0,0.0,0.0};   // maximum boundary
    int ierrflag=0;
    double arglist[10];
    arglist[0]=1;


    //  Let minuit know about the parameters.
    minuit->mnparm(0,"cent", vstrt[0], stp[0],bmin[0],bmax[0],ierrflag);
    minuit->mnparm(1,"sigma", vstrt[1], stp[1],bmin[1],bmax[1],ierrflag);
    minuit->mnparm(2,"norm", vstrt[2], stp[2],bmin[2],bmax[2],ierrflag);

    // initialize Minuit.
    minuit->mnexcm("SET STR",arglist,1,ierrflag);

    // Set this to be 1 for chi square fits
    // Set this to be 0.5 for maximum likelihood fits
    // From the Minuit manual:
    // Sets the value of UP (default value= 1.), defining parameter errors.
    // Minuit defines parameter errors as the change in parameter value required to
    // change the function value by UP.
    // Normally, for chisquared fits UP=1, and for negative log likelihood, UP=0.5
    arglist[0] = 0.5;
    gMinuit->mnexcm("SET ERR", arglist ,1,ierrflag);


    // Which parameters do you want fixed/free.  
    //minuit->FixParameter(0);
    //minuit->FixParameter(1);
    //minuit->FixParameter(2);

    //minuit->SetMaxIterations(500);
    // This does the fit.
    minuit->SetPrintLevel(1);

    minuit->Migrad();    
    double newx, newy, news, newserr;

    // get the new parameters from minuit.
    minuit->GetParameter(2,news, newserr);
    minuit->GetParameter(1,newy, newserr);
    minuit->GetParameter(0,newx, newserr);

    clock_t stop = clock() ;

    float time = (stop - start)/(1.*CLOCKS_PER_SEC);

    std::cout<<"Time ="<<time<<std::endl;

    /*
       std::cout<<"GPU total "<<sum<<std::endl;

       double total = 0.;
       for (int i = 0; i< N; i++){
       float x = vars[i];
       double val = (1./(sqrt(TWO_PI)*sigmaval))*exp(-1.*(centval-x)*(centval-x)/(2.*sigmaval*sigmaval));
    //if(fabs(result[i] - val)>0.0000001){
    // std::cout<<result[i]<<" "<<val<<std::endl;
    //}
    total+=val;
    }

    std::cout<<"CPU total "<<total<<std::endl;
     */
} 
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
