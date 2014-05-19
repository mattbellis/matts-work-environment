#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <ctime>
#include <iostream>
#include <iterator>
#include <cstdlib>
#include "TRandom.h"
#include "TMinuit.h" 

// global variables suck
const size_t N = 1<<14; // 31<<20; //=1048576
thrust::host_vector<float> vars(N) ;
TMinuit* minuit = 0;
TRandom r;
//thrust::device_vector<float> d_vars;// = vars;

// Make Random number
float make_random_float(void)
{
    return r.Gaus();
    //return  static_cast<float>(min(RAND_MAX-1,rand())) / (RAND_MAX);
}

// We'll use a 5-tuple to store our data and variables
typedef thrust::tuple<float,float,float,float,float> Float5;

// This is "the kernel"
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

            thrust::get<0>(a) = log((1./(sqrt(6.2831853)*sigma))*norm*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));

            //return 1./(sqrt(6.2831853)*sigma)*exp(-1.*(cent-x)*(cent-x)/(2.*sigma));

        }
};

void gpufcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    //f = ParFunc(par[0],par[1],par[2],5);
    // for this call to the function
    float norm = par[2];
    float sigmaval = par[1];
    float centval = par[0];


    thrust::device_vector<float> d_vars= vars;

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
    //std::cout<<"Starting"<<std::endl;

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


    //thrust::device_vector<float> normvector(N); //=	(1./(sqrt(6.2831853)*sigma))*norm
    //thrust::fill(normvector.begin(), nomrvector.end(),(1./(sqrt(6.2831853)*sigma))*norm);
    //thrust::device_vector<float> terms(N);
    //thrust::transform(result.begin(), result.end(), normvector.begin(), terms.begin(),
    //thrust::multiplies<float>());	


    // sum it up
    float sum = thrust::reduce(result.begin(), result.end());

    f= (double) 2.*(-1.*sum + norm);
    std::cout<<"FCN="<<f<<":  norm="<<norm<<";  sigma = "<<sigmaval<<";  cent="<<centval<<std::endl;
} 


void cpufcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
    double norm = par[2];
    double sigma = par[1];
    double cent = par[0];

    double runningsum=0;

    for(int i = 0; i<N; i++){
        double x = vars[i];
        runningsum += log(norm*(1./(sqrt(6.2831853)*sigma))*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma)));
    }

    f = 2.*(-1.*runningsum + norm);
    std::cout<<"FCN="<<f<<":  norm="<<norm<<";  sigma = "<<sigma<<";  cent="<<cent<<std::endl;
}




int main(void)
{
    // number of datapoints
    //const size_t N = 1<<24; // 31<<20; //=1048576

    std::cout<<"N objects:"<<N<<std::endl;

    // generate the mock data.  I think this is actually on the CPU

    thrust::generate(vars.begin(), vars.end(), make_random_float);  
    d_vars = vars;
    //for(int i = 0; i< N; i++){
    //	std::cout<<vars[i]<<std::endl;
    //}

    // move it to the GPU (now in the fcn)

    clock_t start = clock() ;

    // Define the minuit object.
    minuit = new TMinuit(3);
    minuit->SetFCN(gpufcn);

    float mean=0.2;
    float sig = 1.2;
    float norm = N;
    double vstrt[3]={mean,sig,norm}; // initial guesses for parameters
    double stp[3]={0.2,0.2,1000}; // step size in the minimization
    double bmin[3]={-5,0,99}; // minimum boundary
    double bmax[3]={5,5,2*N};   // maximum boundary
    int ierrflag=0;
    double arglist[10];
    arglist[0]=1;


    //  Let minuit know about the parameters.
    minuit->mnparm(0,"cent", vstrt[0], stp[0],bmin[0],bmax[0],ierrflag);
    minuit->mnparm(1,"sigma", vstrt[1], stp[1],bmin[1],bmax[1],ierrflag);
    minuit->mnparm(2,"norm", vstrt[2], stp[2],bmin[2],bmax[2],ierrflag);

    // initialize Minuit.
    minuit->mnexcm("SET STR",arglist,1,ierrflag);

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
       double val = (1./(sqrt(6.2831853)*sigmaval))*exp(-1.*(centval-x)*(centval-x)/(2.*sigmaval*sigmaval));
    //if(fabs(result[i] - val)>0.0000001){
    // std::cout<<result[i]<<" "<<val<<std::endl;
    //}
    total+=val;
    }

    std::cout<<"CPU total "<<total<<std::endl;
     */
} 
