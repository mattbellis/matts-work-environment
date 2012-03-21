//
// GPU implementation
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "cutil.h"

/********************************************************/


__global__ void CalcSep(float* raA, float* decA, int ngals)
{
  //does all the i's simultaneously - one for each thread 
  int ix = blockDim.x * blockIdx.x + threadIdx.x;

  float sep=0;
  // Do 1 ``column"
  for(int ij=ix+1;ij<ngals;ij++)
    {
      sep = acos( sin(decA[ix])*sin(decA[ij]) + \
            cos(decA[ix])*cos(decA[ij])*cos(fabs(raA[ix]-raA[ij])) );
    }//loop over gals

  // Then the ngals-ix ``column"
  ix = (ngals - 1) - ix;
  for(int ij=ix+1;ij<ngals;ij++)
    {
      sep = acos( sin(decA[ix])*sin(decA[ij]) + \
            cos(decA[ix])*cos(decA[ij])*cos(fabs(raA[ix]-raA[ij])) );
    }//loop over gals
}


double execute_kernel_gpu(int ngals){

//for some reason Size_t doesn't work here? 
  const int sizeneededin = ngals * sizeof(float);

  //allocate vectors in host memory
  float* h_raA = (float*)malloc(sizeneededin);
  float* h_decA = (float*)malloc(sizeneededin);
  srand(time(0));

  //initailise input vectors - place galaxies at rando coords between 0 and 1
  for(int i=0;i<ngals;i++)
    {
      h_raA[i] = rand(); 
      h_decA[i] = rand();
    }

  //allocate vectors in device memory
  float* d_raA;  float* d_decA; 
  cudaMalloc(&d_raA, sizeneededin);
  cudaMalloc(&d_decA, sizeneededin);
 
  //copy vectors from host to device memory 
  cudaMemcpy(d_raA, h_raA, sizeneededin, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_decA, h_decA, sizeneededin, cudaMemcpyHostToDevice); 

  //invoke kernel
//nthreads: doing half of ngals*ngals matrix. Plus, each thread is doing 2 calcs. 
  int threadsPerBlock = (ngals*ngals)/4; //128;
        // Only need 1/2 as many threads
//  int blocksPerGrid = (ngals/2 + threadsPerBlock -1) / threadsPerBlock; //???????
int blocksPerGrid = 1;

  //set up the cuda timer. 
  //can't use simple CPU timer since that would only time the kernel launch overhead. 
  // Need to make sure all threads have finished before stop the timer - so can synchronise threads before and after kernel launch if using cpu timer? I didn't get sensible results when I've tried that though. 

  cudaEvent_t cudastart, cudaend;
  cudaEventCreate(&cudastart); 
  cudaEventCreate(&cudaend);
  //record the start time
  cudaEventRecord(cudastart,0);
  //run the kernel! 
  CalcSep<<<blocksPerGrid, threadsPerBlock>>>(d_raA, d_decA, ngals);
  //record the end time
  cudaEventRecord(cudaend,0);
  cudaEventSynchronize(cudaend);

  //how long did the kernel take? this gives time in ms
  float cudaelapsed=0;
  cudaEventElapsedTime(&cudaelapsed, cudastart, cudaend);
  printf("elapsed time for GPU in ms: %f",cudaelapsed);
  printf("\n");

  //delete memory
  cudaEventDestroy(cudastart);
  cudaEventDestroy(cudaend);

  //free device memory
  cudaFree(d_raA); cudaFree(d_decA);
  //free host memory
  free(h_raA); free(h_decA); 

return cudaelapsed;

}
