#include<stdio.h>

//__device__ float device_memory[16];
//__shared__ float device_memory[16];

#define NBINS 16
#define NTHREADS_PER_BLOCK 16

__global__ void kernel_hist (uint *partial_hists)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[NBINS];
    //uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;


    // Clear out the shared memory
    //for (int i=0;i<NBINS;i++)
    //{
        //s_Hist[threadIdx.x*NTHREADS_PER_BLOCK + i] = 0.0;
        s_Hist[threadIdx.x] = 0.0;
    //}
    __syncthreads();

    uint local_hist[NBINS];
    // Clear out the shared memory
    for (int i=0;i<NBINS;i++)
    {
        local_hist[i] = 0.0;
    }

    // Compute the index variable
    //int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Add some information to each bin
    for (int i=0;i<NBINS;i++)
    {
        for (int j=0;j<10;j++)
        {
            local_hist[i] += 1;
        }
    }
    __syncthreads();

    for (int i=0;i<NBINS;i++)
    {
        partial_hists[threadIdx.x] += local_hist[i];
    }
}

int main()
{
    int nbins = 16;
    int dimx = 16;
    int num_bytes = dimx*sizeof(uint);

    uint *d_a=0, *h_a=0; // device and host pointers

    // Allocate memory on host (CPU)
    h_a = (uint*)malloc(num_bytes);

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_a,num_bytes);

    // Check to see that there was enough memory for both 
    // allocations.
    // If the memory allocation fails, it doesn't change the 
    // pointer value. That is why we set them to be 0 at declaration,
    // and then see if they have changed or stayed the same. 
    if (0==h_a || 0==d_a)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }

    // Initialize array to all 0's
    cudaMemset(d_a,0,num_bytes);

    //-----------------------------------------------------------------------//
    // Some explanatory code
    /*
    // This will give us 256 thread blocks, arranged in a 16x16 grid.
    dim3 grid(16,16);

    // This will give us 256 threads/block, arranged in a 16x16 grid.
    dim3 block(16,16);

    kernel<<<grid,block,0,0>>>(XXX);

    // This is a shortcut for launching some thread blocks.
    // It launches a grid of 32 thread blocks arranged in a 1x32 grid
    // and 512 threads per block, arranged in a 1x512 array.
    kernel<<<32,512>>>(YYY);
    */

    //dim3 grid,block;
    //block.x = 8;
    //grid.x = dimx/block.x;
    //kernel_hist<<<grid,block>>>(d_a);
    kernel_hist<<<64,16>>>(d_a);

    //dim3 grid(16,16);
    //dim3 block(16,16);
    //kernel_hist<<<grid,block,0,0>>>(d_a);
    //cudaThreadSynchronize();

    // Copy it back over
    cudaMemcpy(h_a,d_a,num_bytes,cudaMemcpyDeviceToHost);

    for (int i=0;i<dimx;i++)
    {
        printf("%d ",h_a[i]);
    }
    printf("\n");


    //printf("\n");
    //for (int i=0;i<dimx;i++)
    //{
        //printf("%f ",device_memory[i]);
    //}
    //printf("\n");


    free(h_a);
    cudaFree(d_a);
    //cudaFree(device_memory);

    return 0;
}
