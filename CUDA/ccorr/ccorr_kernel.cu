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
 * Device code.
 */

#ifndef _ccorr_KERNEL_H_
#define _ccorr_KERNEL_H_

#include <stdio.h>
#include <math.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_raA, float* g_decA, float* g_raB, float* g_decB, int* g_odata) 
{

  // access thread id
  const unsigned int tid_x = threadIdx.x;
  // access number of threads in this block
  const unsigned int block_x = blockIdx.x;
  const unsigned int bsize_x = blockDim.x;
  const int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
  
  
  // access thread id
  const unsigned int tid_y = threadIdx.y;
  // access number of threads in this block
  const unsigned int block_y = blockIdx.y;  
  const unsigned int bsize_y = blockDim.y;
  const int gid_y = blockDim.y*blockIdx.y + threadIdx.y;
  

  // shared memory
  // the size is determined by the host application
  __shared__  float s_raA[16];
  __shared__  float s_decA[16];
  __shared__  float s_raB[16];
  __shared__  float s_decB[16];

  
  s_raA[tid_x]  = g_raA[gid_x];
  s_decA[tid_x] = g_decA[gid_x];
  s_raB[tid_y]  = g_raB[gid_y];
  s_decB[tid_y] = g_decB[gid_y];
  
  //__syncthreads();
  
  __shared__  unsigned int s_res[16][16];

  //s_res[tid_y][tid_x]=0;
  //__syncthreads();	
  
  float sep = acos( sin(s_decA[tid_x])*sin(s_decB[tid_y]) + cos(s_decA[tid_x])*cos(s_decB[tid_y])*cos(fabs(s_raA[tid_x]-s_raB[tid_y])) );
  if(sep<0){sep=0;}
  s_res[tid_y][tid_x] = int(floor((sep/3.2)*64.));

  //s_res[bsize_x*tid_y + tid_x] = int(floor((sep/3.2)*64.));

  //__syncthreads();	

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation
	//SDATA(tid) = g_idata[tid];
  //__syncthreads();

  // perform some computations
  //SDATA(tid) = (float) num_threads * SDATA( tid);
  //__syncthreads();

  // write data to global memory
  //g_odata[(block_y*bsize_y+tid_y)*gridDim.x+(block_x*bsize_x+tid_x)] = s_res[tid_x][tid_y];//s_res[bsize_x*tid_y + tid_x];

  //g_odata[(gid_y*blockDim.y*gridDim.y)+(gid_x)] = s_res[tid_x][tid_y];

	//if( tid_x + tid_y < 1024*1024)
  //g_odata[1024*(blockIdx.y*blockDim.y+tid_y) + (blockIdx.x*blockDim.x+tid_x)] =s_res[tid_x][tid_y];//gid_x;
  
  __shared__ int s_hist[64];
 
  
  if(tid_y*bsize_x + tid_x < 64)
		s_hist[tid_y*bsize_x + tid_x]=0;

  __syncthreads();


  int thisbin=tid_y*bsize_x + tid_x;
  int count=0;
  for(int ival=0; ival<16; ++ival){
	for(int jval = 0; jval<16; ++jval){
		if(thisbin==s_res[ival][jval]) count++;
		__syncthreads();
	}	
  }	

	

  //
 // {	
	s_hist[thisbin]=count;
	if(thisbin< 64)
		g_odata[(blockIdx.y*gridDim.x + blockIdx.x)*64+thisbin] = s_hist[thisbin];
	
   
}

#endif // #ifndef _ccorr_KERNEL_H_
