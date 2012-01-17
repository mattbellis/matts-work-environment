#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <iostream>
#include <iterator>
#include <cstdlib>



// Make Random number
float make_random_float(void)
{
  return  static_cast<float>(min(RAND_MAX-1,rand())) / (RAND_MAX);
}

// We'll use a 3-tuple to store our data and variables
typedef thrust::tuple<float,float,float,float> Float4;

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

    thrust::get<0>(a) = (1./(sqrt(6.2831853)*sigma))*exp(-1.*(cent-x)*(cent-x)/(2.*sigma*sigma));
    //return 1./(sqrt(6.2831853)*sigma)*exp(-1.*(cent-x)*(cent-x)/(2.*sigma));
    
  }
};


int main(void)
{
  // number of datapoints
  const size_t N = 1<<24; // 31<<20; //=1048576

  // generate the mock data.  I think this is actually on the CPU
  thrust::host_vector<float> vars(N) ;
  thrust::generate(vars.begin(), vars.end(), make_random_float);  

  

  // move it to the GPU
  thrust::device_vector<float> d_vars = vars;

  // for this call to the function
  float sigmaval = 0.3;
  float centval = 1.;

  // set all of these vectors to this.
  thrust::device_vector<float> d_sigma(N,sigmaval);
  thrust::device_vector<float> d_cent(N, centval);
  // Another way to do this:
  //  thrust::device_vector<float> d_sigma(N);
  //  thrust::fill(d_sigma.begin(), d_sigma.end(), sigmaval);
  //

  //result vector
  thrust::device_vector<float> result(N);

  // do it.
  std::cout<<"Starting"<<std::endl;
  
  thrust::for_each(thrust::make_zip_iterator(make_tuple(result.begin(),
							 d_vars.begin(), 
							 d_sigma.begin(), 
							 d_cent.begin())),
		    thrust::make_zip_iterator(make_tuple(result.end(),
							 d_vars.end(), 
							 d_sigma.end(), 
							 d_cent.end())),
		    Fcn() );
  
  

  // sum it up
  float sum = thrust::reduce(result.begin(), result.end());

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
}
