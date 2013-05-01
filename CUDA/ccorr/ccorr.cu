/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <ccorr_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( int* reference,  float* iraA, float* idecA, float* jraB, float* jdecB, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    //cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
  

	int ngals=2048;
	//int ngals=128;

    unsigned int num_threads = ngals*ngals;
    unsigned int mem_size = sizeof( float) * ngals;

    // allocate host memory
    float* h_raA = (float*) malloc( mem_size);
	float* h_decA = (float*) malloc( mem_size);
	float* h_raB = (float*) malloc( mem_size);
	float* h_decB = (float*) malloc( mem_size);
	
    // initalize the memory
    for( unsigned int i = 0; i < ngals; ++i) 
    {
        h_raA[i]  = 3.14*((float)(rand()%1000))/1000.;
		h_decA[i] = 6.28*((float)(rand()%1000))/1000.;
		h_raB[i]  = 3.14*((float)(rand()%1000))/1000.;
		h_decB[i] = 6.28*((float)(rand()%1000))/1000.;
    }


	cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // allocate device memory
	float* d_raA;
	float* d_decA;
	float* d_raB;
	float* d_decB;
	
    cutilSafeCall( cudaMalloc( (void**) &d_raA, mem_size));
	cutilSafeCall( cudaMalloc( (void**) &d_decA, mem_size));
	cutilSafeCall( cudaMalloc( (void**) &d_raB, mem_size));
	cutilSafeCall( cudaMalloc( (void**) &d_decB, mem_size));
	
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_raA, h_raA, mem_size,
                                cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy( d_decA, h_decA, mem_size,
                                cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy( d_raB, h_raB, mem_size,
                                cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy( d_decB, h_decB, mem_size,
                                cudaMemcpyHostToDevice) );



    // allocate device memory for result
    int* d_odata;	
    cutilSafeCall( cudaMalloc( (void**) &d_odata, sizeof(int)*64*128*128));
	// allocate mem for the result on host side
	int* h_odata = (int*) malloc( sizeof(int)*64*128*128);



    // setup execution parameters
    dim3  grid( 128, 128);
    dim3  threads( 16, 16);

	size_t shared_mem_size = (sizeof(int)*16*16)+ (sizeof(int)*64) + (sizeof(float)*4*32);
	printf("Allocating %i shared memory\n",shared_mem_size);
	size_t return_mem_size = (sizeof(int)*ngals*ngals);
	printf("Allocating %i global memory\n",return_mem_size);
	

		//clock_t  t1 = clock();
		
		
	cudaEvent_t cudastart, cudaend;
    cudaEventCreate(&cudastart); 
    cudaEventCreate(&cudaend);
	//record the start time
    cudaEventRecord(cudastart,0);
	
		
    // execute the kernel
    //testKernel<<< grid, threads, shared_mem_size >>>( d_raA, d_decA, d_raB, d_decB, d_odata);
	testKernel<<< grid, threads >>>( d_raA, d_decA, d_raB, d_decB, d_odata);

	
	cudaEventRecord(cudaend,0);
    cudaEventSynchronize(cudaend);

	
	float cudaelapsed=0;
    cudaEventElapsedTime(&cudaelapsed, cudastart, cudaend);
    printf("elapsed time for GPU in ms: %f\n",cudaelapsed);
	
	
	// check if kernel execution generated an error
    cutilCheckMsg("Kernel execution failed");
	// copy result from device to host
	cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof(int)*64*128*128, cudaMemcpyDeviceToHost) ) ;

	

	cutilCheckError( cutStopTimer( timer));
    //printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));



	int array[64];
	for(int i = 0; i< 64; i++)
	{
		array[i]=0;
	}
	for(int ientry=0; ientry<64*128*128; ientry++)
	{
			array[ientry%64]+=h_odata[ientry];		
	}	
	std::cout<<"out"<<std::endl;
    // compute reference solution
    int* reference = (int*) malloc(64*sizeof(int));
	std::cout<<"Computing CPU"<<std::endl;
	
	clock_t time=clock();
	
	computeGold( reference, h_raA, h_decA, h_raB, h_decB, ngals);

	clock_t testend = clock();
	std::cout<<"CPU: "<< ((testend-time)/(1.*CLOCKS_PER_SEC))*1000. <<std::endl;
	

    // check result
    //if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    //{
        // write file for regression test
	//cutilCheckError( cutWriteFilef( "./data/regression.dat",h_odata, ngals, 0.0));
    //}
    //else 
    //{
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
	printf("checking\n");
	int countGPU=0;
	int countCPU=0;
	for(int k = 0; k< 64; k++){
		std::cout << reference[k] << " " << array[k] << std::endl;
		countCPU+=reference[k];
		countGPU+=array[k];	
	}
	
	std::cout <<countCPU<< " " << countGPU << std::endl;
	
	//CUTBoolean res = cutComparei( reference, h_odata, num_threads);
	//printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    //}

    // cleanup memory
    free( h_raA);
	free( h_decA);
	free( h_raB);
	free( h_decB);
	
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_raA));
	cutilSafeCall(cudaFree(d_decA));
	cutilSafeCall(cudaFree(d_raB));
	cutilSafeCall(cudaFree(d_decB));
  
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}
