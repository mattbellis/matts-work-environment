#include<stdio.h>

//#define NUM_BINS 64 // We are also going to use this for the number of threads in a block.
#define NUM_BINS 1024 // We are also going to use this for the number of threads in a block.

#define NUM_THREADS_PER_BLOCK 16
#define NUM_BLOCKS 16

////////////////////////////////////////////////////////////////////////////////
// Took this code from a Dr. Dobbs example.
////////////////////////////////////////////////////////////////////////////////
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
// The kernel. 
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel (int *dev_hist)
{

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    __shared__ int shared_hist[NUM_BINS];

    // access thread id
    //const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // Note that we only clear things out for the first thread on each block.
    if(threadIdx.x==0)
    {
        for (int i=0;i<NUM_BINS;i++)
            shared_hist[i] = 0;
    }
    __syncthreads();
    ////////////////////////////////////////////////////////////////////////

    // FILL THE ARRAYS

    shared_hist[tid] = blockIdx.x*1000 + tid;
    //shared_hist[tid] = blockIdx.x;

    __syncthreads();

    if(threadIdx.x==0)
    {
        for(int i=0;i<NUM_BINS;i++)
        {
            dev_hist[i+(blockIdx.x*(NUM_BINS))]=shared_hist[i];
        }
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// The main() program.
////////////////////////////////////////////////////////////////////////////////
int main()
{
    int nbins = NUM_BINS;
    int dimx = NUM_BLOCKS; // Number of blocks
    int num_bytes = NUM_BINS*sizeof(int);
    int num_bytes_on_gpu = dimx*NUM_BINS*sizeof(int);

    int *d_a=0, *h_a=0, *h_hist=0; // device and host pointers

    // Allocate memory on host (CPU)
    h_a = (int*)malloc(num_bytes_on_gpu);
    h_hist = (int*)malloc(num_bytes);

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_a,num_bytes_on_gpu);
    checkCUDAError("malloc");

    // Check to see that there was enough memory for both 
    // allocations.
    // If the memory allocation fails, it doesn't change the 
    // pointer value. That is why we set them to be 0 at declaration,
    // and then see if they have changed or stayed the same. 
    if (0==h_a)
    {
        printf("Couldn't allocate host memory\n");
        return 1;
    }

    if (0==d_a)
    {
        printf("Couldn't allocate device memory\n");
        return 1;
    }

    // Initialize array to all 0's
    cudaMemset(d_a,0,num_bytes_on_gpu);
    checkCUDAError("memset");

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

    dim3 grid, block;
    grid.x = NUM_BLOCKS; // Number of blocks
    block.x = NUM_BINS; // Number of threads per block.
    
    kernel<<<grid,block>>>(d_a);
    cudaThreadSynchronize();
    checkCUDAError("kernel");

    // Copy it back over
    cudaMemcpy(h_a,d_a,num_bytes_on_gpu,cudaMemcpyDeviceToHost);

    for (int i=0;i<dimx;i++)
    {
        for (int j=0;j<NUM_BINS;j++)
        {
            printf("%d ",h_a[i*NUM_BINS + j]);
        }
        printf("\n");
    }
    printf("\n");


    free(h_a);
    cudaFree(d_a);

    return 0;
}
