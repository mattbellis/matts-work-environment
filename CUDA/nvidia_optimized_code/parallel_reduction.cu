#include<stdio.h>
#include<iostream>
#include <stdlib.h>
#include <algorithm>

#define MAX_BLOCK_DIM_SIZE 65535

using namespace std;

__global__ void reduce(int *g_idata, int *g_odata, int num_bytes) {
	
	// create shared memory array
	extern __shared__ int sdata[];	

	// each thread loads one element from global to shared mem	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	// sync threads 
	__syncthreads();
	
	
	// do reduction in shared mem
	// s = 1,2,4,8,...
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		// if tid is multiple of 2,4,8,....
		if (tid % (2*s) == 0) {
			// ie: sdata[4] += sdata[4+2]
			//   sdata[6] += sdata[6+2]
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// now they are all added up...
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}




////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#define MIN(x,y) ((x < y) ? x : y)


void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    
    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
        

    if (whichKernel == 6)
        blocks = MIN(maxBlocks, blocks);
}



int main()
{
	// Number of values to be added together / averaged / whatever.
	int Nvals = 1<<18; //pow(2,22);
	
	// Number of Threads
	int maxThreads = 256;
	//int Threads = 256;//Nvals;//256;
	//int numThreads = 256;//Nvals;//256;
	
	// Number of Blocks
	int maxBlocks = MIN( Nvals / maxThreads, MAX_BLOCK_DIM_SIZE);//64;
	//int numBlocks = 64;
	

	int num_bytes = Nvals*sizeof(int);
		
	// Allocate memory on host (CPU)
	int* h_idata = (int*)malloc(num_bytes);
	int* h_odata = (int*)malloc(num_bytes);

	int* d_idata=0;
	int* d_odata=0;	
	
	// Allocate memory on device (GPU)
    cudaMalloc((void**)&d_idata,num_bytes);
	cudaMalloc((void**)&d_odata,num_bytes);


	// Check to see that there was enough memory for both 
    // allocations.
    // If the memory allocation fails, it doesn't change the 
    // pointer value. That is why we set them to be 0 at declaration,
    // and then see if they have changed or stayed the same. 
    if (0==h_idata || 0==d_odata)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }

	// Initialize array to all 0's
    cudaMemset(d_idata,0,num_bytes);
	cudaMemset(d_odata,0,num_bytes);


	// Let's create random numbers for input on CPU
	int sum=0;
	for(int i=0; i<Nvals; i++) {
		h_idata[i] = (int)(rand() % 100);
		sum += h_idata[i];
		//std::cout<<h_idata[i]<<std::endl;
	}

	

	// copy it over
	cudaMemcpy(d_idata,h_idata,num_bytes,cudaMemcpyHostToDevice);

	int s=Nvals;
	int threads=s;
	int blocks=256;
	
//	for(int j = 4; j>0; j--){
//		if(j==4) s=Nvals;
//		if(j==3) s=Nvals/(1<<6);
//		if(j==2) s=Nvals/(1<<12);
//		if(j==2) s=Nvals/(1<<18);
					
	//	getNumBlocksAndThreads(6, s, maxBlocks, maxThreads, blocks, threads);
	
		blocks=256;
		threads = blocks / 2;
				
		cout<<blocks<<" Blocks, "<<threads<<" Threads"<<endl; 
		
		// Still don't really know what I'm doing here 	
		dim3 grid,block;
		block.x = blocks;
		// block.y = 4;
		grid.x = Nvals/block.x;
		//grid.y = NPTS/block.y;
	
	
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

		reduce<<< grid, block, smemSize >>>(d_idata, d_odata, num_bytes);

		block.x = 4;
		Nvals = Nvals / blocks;
		grid.x = Nvals/block.x;	
		
		
		

//	}
	// copy it back
	cudaMemcpy(h_odata,d_odata,num_bytes,cudaMemcpyDeviceToHost);

	for(int i=0; i<Nvals; i++) {
		if(h_odata[i] == 0) continue; 
			else{std::cout <<i<<" "<<h_odata[i]<<std::endl;}			
	}


	printf("%d \n",h_odata[0]);
	
	
	printf("Should be %d \n",sum);

	

 
	return 0;		

}
