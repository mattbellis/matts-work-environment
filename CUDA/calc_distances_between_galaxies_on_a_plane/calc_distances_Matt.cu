#include<stdio.h>
#include<stdlib.h>
#include<cmath>

int NPTS = 20000;

__global__ void kernel (float *a, float *b, int dimx, int dimy)
{
    // Compute the index variable
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    int idx = iy*dimx + ix;

    //a[idx] = a[idx]+1;
    float r, xdiff, ydiff;
    for (int i=idx;i<dimx;i++)
    {
        if (i != idx)
        {
            xdiff = a[idx] - a[i];
            ydiff = b[idx] - b[i];
            r = sqrt(xdiff*xdiff + ydiff*ydiff);
        }
    }

}

int main()
{

    float xmax = 10.0;
    float ymax = 10.0;

    int num_bytes = NPTS*sizeof(float);

    float *d_x=0, *d_y=0, *h_x=0, *h_y=0; // device and host pointers

    // Allocate memory on host (CPU)
    h_x = (float*)malloc(num_bytes);
    h_y = (float*)malloc(num_bytes);

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_x,num_bytes);
    cudaMalloc((void**)&d_y,num_bytes);


    // Check to see that there was enough memory for both 
    // allocations.
    // If the memory allocation fails, it doesn't change the 
    // pointer value. That is why we set them to be 0 at declaration,
    // and then see if they have changed or stayed the same. 
    if (0==h_x || 0==d_x || 0==h_y || 0==d_y)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }
    
    // Fill the universe with random stuff
    for (int i=0;i<NPTS;i++)
    {
        h_x[i] = 2.0*xmax*rand() - xmax;
        h_y[i] = 2.0*ymax*rand() - ymax;
    }


    // Initialize array to all 0's
    cudaMemset(d_x,0,num_bytes);
    cudaMemset(d_y,0,num_bytes);

    //-----------------------------------------------------------------------//

    dim3 grid,block;
    block.x = 4;
    block.y = 4;
    grid.x = NPTS/block.x;
    grid.y = NPTS/block.y;

    cudaMemcpy(d_x,h_x,num_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,num_bytes,cudaMemcpyHostToDevice);

    kernel<<<grid,block>>>(d_x,d_y,NPTS,NPTS);

    /*
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
    */

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
