#include<stdio.h>

__global__ void kernel (int *a, int dimx, int dimy)
{
    // Compute the index variable
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    int idx = iy*dimx + ix;

    //a[idx] = a[idx]+1;
    a[idx] = iy*dimx + ix;

}

int main()
{
    int dimx = 16;
    int dimy = 16;

    int num_bytes = dimy*dimx*sizeof(int);

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

    dim3 grid,block;
    block.x = 4;
    block.y = 4;
    grid.x = dimx/block.x;
    grid.y = dimx/block.y;

    kernel<<<grid,block>>>(d_a,dimx,dimy);

    // Copy it back over
    cudaMemcpy(h_a,d_a,num_bytes,cudaMemcpyDeviceToHost);

    for (int row=0;row<dimy;row++)
    {
        for (int col=0;col<dimx;col++)
        {
            printf("%d",h_a[row*dimx+col]);
        }
        printf("\n");
    }

    free(h_a);
    cudaFree(d_a);

    return 0;
}
