#include<stdio.h>

__global__ void kernel (int *a)
{
    // Compute the index variable
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //a[idx] = 7;

    // Try some other possible memory applications

    // Set equal to the block Idx
    //a[idx] = blockIdx.x;

    // Set equal to the thread Idx
    //a[idx] = threadIdx.x;
    // Set equal to the Idx
    a[idx] = idx;
}

int main()
{
    int dimx = 16;
    int num_bytes = dimx*sizeof(int);

    int *d_a=0, *h_a=0; // device and host pointers

    // Allocate memory on host (CPU)
    h_a = (int*)malloc(num_bytes);

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

    dim3 grid,block;
    block.x = 4;
    grid.x = dimx/block.x;

    kernel<<<grid,block>>>(d_a);

    // Copy it back over
    cudaMemcpy(h_a,d_a,num_bytes,cudaMemcpyDeviceToHost);

    for (int i=0;i<dimx;i++)
    {
        printf("%d\n",h_a[i]);
    }
    printf("\n");

    free(h_a);
    cudaFree(d_a);

    return 0;
}
