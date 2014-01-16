#include<stdio.h>
#include<stdlib.h>
#include<cmath>

int main()
{
    int npts = 20000;
    //int dimx = 16;
    //int dimy = 16;

    float xmax = 10.0;
    float ymax = 10.0;

    int num_bytes = npts*sizeof(float);

    //int *d_a=0, *h_a=0; // device and host pointers

    // Allocate memory on host (CPU)
    //h_a = (int*)malloc(num_bytes);
    float *x = (float*)malloc(num_bytes);
    float *y = (float*)malloc(num_bytes);


    for (int i=0;i<npts;i++)
    {
        x[i] = 2.0*xmax*rand() - xmax;
        y[i] = 2.0*ymax*rand() - ymax;
    }


    float xdiff,ydiff,r;

    for (int i=0;i<npts-1;i++)
    {
        for (int j=i+1;j<npts;j++)
        {
            xdiff = x[i]-x[j];
            ydiff = y[i]-y[j];
            r = sqrt(xdiff*xdiff + ydiff*ydiff);
        }
    }

    free(x);
    free(y);

    return 0;
}
