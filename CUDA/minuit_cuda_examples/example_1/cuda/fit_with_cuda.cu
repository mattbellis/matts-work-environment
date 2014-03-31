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

#include <fstream>

// global variables suck
const size_t N = 1<<17; // 31<<20; //=1048576
const size_t N_norm = 1<<17; // 31<<20; //=1048576

const float TWO_PI = 6.2831853071795862;

thrust::host_vector<float> vars(N) ;

thrust::host_vector<float> xvars(N) ;
thrust::host_vector<float> yvars(N) ;
thrust::host_vector<float> zvars(N) ;

thrust::host_vector<float> norm_vars(N_norm) ;
TMinuit* minuit = 0;
TRandom3 r;

thrust::device_vector<float> d_vars;// = vars;

thrust::device_vector<float> d_xvars;// = vars;
thrust::device_vector<float> d_yvars;// = vars;
thrust::device_vector<float> d_zvars;// = vars;


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
// This is "the kernel"
///////////////////////////////////////////////////////////////////////////////
struct Fcn {
    template <typename Tuple>
        __host__ __device__
        void operator()(Tuple a)
        {
            // dummy Likelihood
            float x = thrust::get<1>(a);  // data
            float y = thrust::get<2>(a);  // data
            float z = thrust::get<3>(a);  // data

            // sig/tot param
            float sig_frac = thrust::get<4>(a);
            float bkg_frac = 1.0-bkg_frac;

            // Bkg params
            float tau0 = thrust::get<5>(a);
            float tau1 = thrust::get<6>(a);
            float tau2 = thrust::get<7>(a);

            float norm0 = thrust::get<8>(a);
            float mu0   = thrust::get<9>(a);
            float sig0  = thrust::get<10>(a);

            float norm1 = thrust::get<11>(a);
            float mu1   = thrust::get<12>(a);
            float sig1  = thrust::get<13>(a);

            float norm2 = thrust::get<14>(a);
            float mu2   = thrust::get<15>(a);
            float sig2  = thrust::get<16>(a);

            //thrust::get<0>(a) = log(norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));
            float valx = bkg_frac*exp(-x/tau0) + sig_frac*norm0*exp(-1.*(mu0-x)*(mu0-x)/(2.*sig0*sig0)); 
            float valy = bkg_frac*exp(-y/tau1) + sig_frac*norm1*exp(-1.*(mu1-y)*(mu1-y)/(2.*sig1*sig1)); 
            float valz = bkg_frac*exp(-z/tau2) + sig_frac*norm2*exp(-1.*(mu2-z)*(mu2-z)/(2.*sig2*sig2)); 
            float totval = valx*valy*valz;
            thrust::get<0>(a) = log(totval);

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
    //float norm = par[2];
    //float sigmaval = par[1];
    //float centval = par[0];
    float sig_frac = par[0];

    // Bkg params
    float tau0 = par[1];
    float tau1 = par[2];
    float tau2 = par[3];

    float norm0 = par[4];
    float mu0   = par[5];
    float sig0  = par[6];

    float norm1 = par[7];
    float mu1   = par[8];
    float sig1  = par[9];

    float norm2 = par[10];
    float mu2   = par[11];
    float sig2  = par[12];

    //thrust::device_vector<float> d_vars= vars;

    // set all of these vectors to this.
    //thrust::device_vector<float> d_sigma(N, sigmaval);
    //thrust::device_vector<float> d_cent(N, centval);
    //thrust::device_vector<float> d_norm(N, norm);

    thrust::device_vector<float> d_par0(N, sig_frac);
    thrust::device_vector<float> d_par1(N, tau0);
    thrust::device_vector<float> d_par2(N, tau1);
    thrust::device_vector<float> d_par3(N, tau2);
    thrust::device_vector<float> d_par4(N, norm0);
    thrust::device_vector<float> d_par5(N, mu0);
    thrust::device_vector<float> d_par6(N, sig0);
    thrust::device_vector<float> d_par7(N, norm1);
    thrust::device_vector<float> d_par8(N, mu1);
    thrust::device_vector<float> d_par9(N, sig1);
    thrust::device_vector<float> d_par10(N, norm2);
    thrust::device_vector<float> d_par11(N, mu2);
    thrust::device_vector<float> d_par12(N, sig2);

    //result vector
    thrust::device_vector<float> result(N);

    // do it.
    thrust::for_each(thrust::make_zip_iterator(make_tuple(result.begin(),
                    d_xvars.begin(), 
                    d_yvars.begin(), 
                    d_zvars.begin(), 
                    d_par0.begin(), 
                    d_par1.begin(),
                    d_par2.begin(),
                    d_par3.begin(),
                    d_par4.begin(),
                    d_par5.begin(),
                    d_par6.begin(),
                    d_par7.begin(),
                    d_par8.begin(),
                    d_par9.begin(),
                    d_par10.begin(),
                    d_par11.begin(),
                    d_par12.begin())),
            thrust::make_zip_iterator(make_tuple(result.end(),
                    d_xvars.end(), 
                    d_yvars.end(), 
                    d_zvars.end(), 
                    d_par0.end(), 
                    d_par1.end(),
                    d_par2.end(),
                    d_par3.end(),
                    d_par4.end(),
                    d_par5.end(),
                    d_par6.end(),
                    d_par7.end(),
                    d_par8.end(),
                    d_par9.end(),
                    d_par10.end(),
                    d_par11.end(),
                    d_par12.end())),
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
int main(int argc, char** argv)
{
    // number of datapoints
    //const size_t N = 1<<24; // 31<<20; //=1048576

    std::cout<<"N objects:"<<N<<std::endl;
    std::cout<<"N_norm objects:"<<N_norm<<std::endl;

     // raw pointer to device memory
    int * raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(int));
    std::cerr<<"Allocated raw_ptr:"<<std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // Read in the data
    ///////////////////////////////////////////////////////////////////////////
    float x,y,z;
    ifstream IN(argv[1]);

    int nevents = 0;
    while(IN >> x)
    {
        IN >> y >> z;
        xvars[nevents] = x;
        yvars[nevents] = y;
        zvars[nevents] = z;
        nevents++;
        if (nevents>=N)
        {
            std::cerr << "Too many values in file!" << std::endl;
            exit(-1);
        }
    }

    std::cerr << "Generated the data events...." << std::endl;

    //////////////////////////////////////////
    // Copy over the events to the device
    //////////////////////////////////////////
    xvars.resize(nevents);
    yvars.resize(nevents);
    zvars.resize(nevents);
    d_xvars.resize(nevents);
    d_yvars.resize(nevents);
    d_zvars.resize(nevents);

    //d_vars = vars;
    d_xvars = xvars;
    d_yvars = yvars;
    d_zvars = zvars;
    std::cerr << "Copied over to the device...." << std::endl;

    //////////////////////////////////////////


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
