/// copied largely from deviceQuery



// std::system includes
#include <memory>
#include <iostream>
#include <math.h>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

  /// what size simulation are we trying to use?
  int nx = 2048;
  int ncalc = nx * nx;
  int gals_per_arcmin = 35; // # galaxies per square arcmin
  float survey_angle = sqrt(12.0); // survey angle in degrees
  int tot_gals = floor( gals_per_arcmin * survey_angle * survey_angle * 36000 );
  int gpu_mem_needed = int(tot_gals * sizeof(float)) * 5; // need to allocate gamma1, gamma2, ra, dec and output. 
  printf("Requirements: %d calculations and %d bytes memory on the GPU \n\n", ncalc, gpu_mem_needed);  


  // now get the info from teh device. 
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
  }
  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");
  else
    printf("Found %d CUDA Capable device(s)\n", deviceCount); 
  
  
  int dev, driverVersion = 0, runtimeVersion = 0;     
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  
    printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", 
	  (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    
    
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
	   deviceProp.maxThreadsDim[0],
	   deviceProp.maxThreadsDim[1],
	   deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
	   deviceProp.maxGridSize[0],
	   deviceProp.maxGridSize[1],
	   deviceProp.maxGridSize[2]);
  


  // does this device have enough capcacity for the calculation? 
  printf("\n*************\n");

  // check memory
  if((unsigned long long) deviceProp.totalGlobalMem < gpu_mem_needed) printf(" FAILURE: Not eneough memeory on device for this calculation! \n");
  else
    { 
      printf("Hurrah! This device has enough memory to perform this calculation\n");
      
      // check # threads
  
      int threadsPerBlock = deviceProp.maxThreadsPerBlock; // maximal efficiency exists if we use max # threads per block. 
      int blocksPerGrid = int(ceil(ncalc / threadsPerBlock)); // need nx*nx threads total
      if(deviceProp.maxThreadsDim[0] >blocksPerGrid) printf("FAILURE: Not enough threads on teh device to do this calculation!\n");
      else 
	{
	  printf("Hurrah! This device supports enough threads to do this calculation\n");
	  // how many kernels can we run at once on this machine? 
	  int n_mem = floor(deviceProp.totalGlobalMem / float(gpu_mem_needed)); 
	  int n_threads = floor(threadsPerBlock * deviceProp.maxThreadsDim[0]*deviceProp.maxThreadsDim[1] / float(ncalc) ); // max # threads possible? 

	  printf("%d %d  \n",  n_threads, deviceProp.maxThreadsDim[0]);

	  int max_kernels = 0;
	  n_mem<n_threads ? max_kernels = n_mem : max_kernels = n_threads; 

	  printf(" you can run %d kernels at a time on this device without overloading the resources \n", max_kernels);
	}
    }

  }// loop over devices


}
