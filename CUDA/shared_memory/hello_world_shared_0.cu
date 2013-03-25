#include<stdio.h>

__global__ void kernel (float *out)
{

    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;


    sdata[tid] = 0.0;
    __syncthreads();

    //extern __shared__ float device_memory[];
    // Compute the index variable
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //device_memory[threadIdx.x] += blockDim.x;
    sdata[tid] += blockDim.x;
    //device_memory[threadIdx.x] += threadIdx.x;

    // Do I need this to get the output?
    // Maybe not
    //__syncthreads();

    //out[threadIdx.x] = blockIdx.x;
    out[tid] = sdata[tid];
}

int main()
{
    int nbins = 16;
    int dimx = 16;
    int num_bytes = dimx*sizeof(float);

    float *d_a=0, *h_a=0; // device and host pointers

    // Allocate memory on host (CPU)
    h_a = (float*)malloc(num_bytes);

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
    //kernel<<<grid,block>>>(d_a);
    //kernel<<<4,16>>>(d_a);

    dim3 grid(16,16);
    dim3 block(16,16);
    kernel<<<grid,block,0,0>>>(d_a);
    cudaThreadSynchronize();

    // Copy it back over
    cudaMemcpy(h_a,d_a,num_bytes,cudaMemcpyDeviceToHost);

    for (int i=0;i<dimx;i++)
    {
        printf("%f ",h_a[i]);
    }
    printf("\n");


    free(h_a);
    cudaFree(d_a);

    return 0;
}
