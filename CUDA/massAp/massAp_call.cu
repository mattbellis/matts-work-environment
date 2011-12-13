#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "cutil.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

// Simple utility function to check for CUDA runtime errors
void checkCUDAerror(const char* msg);


//device code
__global__ void doCalc(float* gamma1, float* gamma2, float* dist, float* ang, float* output)
{
  //does all the i's simultaneously - one for each thread 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // each thread has the gamma1 and gamma2 values for one point in the matrix
  // also the distance of this point from the point it's contributing to
  // and the angle from this point. These could be calculated on the GPU, 
  // but are not. Because it's easier to pass the info. Might be quicker to calc tho. 

  //this is a rather complicated calculation. 
  float thetaMax = 32; //chould be passed to fn
  float xc = 0.15; // could be passed to fn
  float x = dist[idx] / thetaMax;
  float Q = (1.0 / (1 + exp(6 - 150*x) + exp(-47 + 50*x))) * (tanh(x/xc) / (x/xc));
  //now for tangential component of shear. 
  float gammaTan = gamma1[idx]*cos(ang[idx]) + gamma2[idx]*cos(ang[idx]+45);
  output[idx] = Q*gammaTan;

}


double execute_kernel_gpu(matrix<float> this_gamma1, matrix<float> this_gamma2){
  /// what am I going to do?
  // This func has been called for one point in the 1024x1024 grid
  // I have the 64x64 matrix of point surrounding this space for g1 and g2
  // I need to go over all of these points, take g1 and g2 for that point
  // and make the calc of the contribution of that point to the total
  // then I can sum them all together at the end and return the variable

  //start with the mem allocation 
  int ncalc = 64*64;
  size_t sizeneeded = ncalc*sizeof(float);
  float *h_gamma1 = 0;
  float *h_gamma2 = 0;
  float *h_dist = 0;
  float *h_ang = 0;
  h_gamma1 = (float*) malloc(sizeneeded);
  h_gamma2 = (float*) malloc(sizeneeded);
  h_dist = (float*) malloc(sizeneeded);
  h_ang = (float*) malloc(sizeneeded);

  //convert the matrices to vectors. GPU can't handle matrices (in this format). 
  int idx=0;
  for(int i=0;i<64;i++){
    for(int j=0;j<64;j++){
      idx = 64*i + j;
      h_gamma1[idx] = this_gamma1(i,j);
      h_gamma2[idx] = this_gamma2(i,j);
      //central point is 32,32
      h_dist[idx] = sqrt(fabs(32-i)*fabs(32-i) + fabs(32-j)*fabs(32-j));
      h_ang[idx] = atan( fabs(32-i)/fabs(32-j)); // this is in radians
      
    }
  }

  //allocate device memory
  float *d_gamma1, *d_gamma2, *d_dist, *d_ang;
  cudaMalloc(&d_gamma1, sizeneeded);
  cudaMalloc(&d_gamma2, sizeneeded);
  cudaMalloc(&d_dist, sizeneeded);
  cudaMalloc(&d_ang, sizeneeded);

  // output vector is going to be the calculated value for each point
  float *h_output, *d_output;
  h_output = (float*)malloc(sizeneeded);
  cudaMalloc(&d_output, sizeneeded);

  //copy vectors from host to device memory
  cudaMemcpy(d_gamma1, h_gamma1, sizeneeded, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma2, h_gamma2, sizeneeded, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dist, h_dist, sizeneeded, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ang, h_ang, sizeneeded, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, sizeneeded, cudaMemcpyHostToDevice);

  //check memory is alright
 if (0==h_gamma1 || 0==h_gamma2 || 0==h_dist || 0==h_ang || 0==h_output) printf("can't allocate memory on host \n");
 if (0==d_gamma1 || 0==d_gamma2 || 0==d_dist || 0==d_ang || 0==d_output) printf("can't allocate memory on device \n");
 checkCUDAerror("memory");
      
 //kernel info - note 512 thread per block max! 
 int threadsPerBlock = 512;
 int blocksPerGrid = 8; // need 64*64 threads total

  doCalc<<<blocksPerGrid, threadsPerBlock>>>(d_gamma1, d_gamma2, d_dist, d_ang, d_output);
 
  checkCUDAerror("kernel");

 //get the output back off the device
 cudaMemcpy(h_output, d_output, sizeneeded, cudaMemcpyDeviceToHost);
 // now sum this up. there is surely a better way to do this..
 double thesum = 0; 
 for(int i=0;i<ncalc;i++){
   if(isnan(h_output[i])) continue;//goes wierd at some point
   thesum +=h_output[i];
 }
 
  //free up the memory
  cudaFree(d_gamma1);
  cudaFree(d_gamma2);
  cudaFree(d_dist);
  cudaFree(d_ang);
  cudaFree(d_output);
  free(h_gamma1);
  free(h_gamma2);
  free(h_dist);
  free(h_ang);
  free(h_output);

  
  // science note: I have to normalise by n points I summed over. 
  return thesum/ncalc;
}

//simple function to check for errors. From Dr Dobbs. 
void checkCUDAerror(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
	      cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


