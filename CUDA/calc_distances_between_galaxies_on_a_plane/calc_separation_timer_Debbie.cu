#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//// notes
// based on the examples given in the CUDE programming guide
// this one makes a list of gals, one list for ra and one for dec. 
// it can then calcs the separation between gal pairs. 
// note that it's not returning anythign from the calculation! 
// just calculating how long each process takes. 
// Try playing with ngals - time scales as you'd expect with CPU
// ans CPU is fatser with fewer gals.  
// then again, this isn't exactly optimised code....
// my numbers:
// 100 gals: 0.6 ms w/ CPU, 13 ms w/ GPU
// 1000 gals: 61 ms w/ CPU, 185 w/ GPU
// 10000 gals: 6085 ms w/ CPU, 4871 w/ GPU

//device code
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


//Host code
int main(int argc, char **argv)
{
  int ngals = 100;

  // Grab the number of galaxies from the command line *if* they have 
  // been specified.
  if (argc>1)
  {
      ngals = atoi(argv[1]);
  }


  size_t sizeneededin = ngals * sizeof(float);

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

  //calculate separation in CPU and calculate time needed
  clock_t teststart = clock();

  float testsep=0;
  for(int i=0;i<ngals;i++){
    for(int j=i+1;j<ngals;j++){
      testsep = acos( sin(h_decA[i])*sin(h_decA[j]) + cos(h_decA[i])*cos(h_decA[j])*cos(fabs(h_raA[i]-h_raA[j])) );
    }
  }
  clock_t testend = clock();
  float testelapsed = (float)(testend-teststart);
  printf("elapsed time for CPU in ms: %f", testelapsed/CLOCKS_PER_SEC*1000);
  printf("\n");


  //allocate vectors in device memory
  float* d_raA;  float* d_decA; 
  cudaMalloc(&d_raA, sizeneededin);
  cudaMalloc(&d_decA, sizeneededin);
 
  //copy vectors from host to device memory 
  cudaMemcpy(d_raA, h_raA, sizeneededin, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_decA, h_decA, sizeneededin, cudaMemcpyHostToDevice); 

  //invoke kernel
  int threadsPerBlock = 256;
  //int threadsPerBlock = 64;
  //int blocksPerGrid = (ngals + threadsPerBlock -1) / threadsPerBlock; //???????
  // Only need 1/2 as many threads
  int blocksPerGrid = (ngals/2 + threadsPerBlock -1) / threadsPerBlock; //???????

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


}
