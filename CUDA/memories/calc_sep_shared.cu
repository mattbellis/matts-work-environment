#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//// notes
// based on the examples given in the CUDE programming guide
// this one makes a list of gals, one list for ra and one for dec. 
// it can then calcs the separation between gal pairs. 
// note that it's not returning anythign from the calculation! 
// just calculating how long each process takes. 

//this version uses shared memory. I'm sticking with 128 gals
//so there's no space issues, but that would be a prob with more. 
//the trick would be to split up the arrays into chunks, 
//then put the chunks into shared mem and have one threadblock
//run over each chunk


//device code
__global__ void CalcSep(float* ra, float* dec, int ngals)
{
    //does all the i's simultaneously - one for each thread 
    int ix = blockDim.x * blockIdx.x + threadIdx.x;

    float sep=0;
    // Do 1 ``column"
    for(int ij=ix+1;ij<ngals;ij++)
    {
        sep = acos( sin(dec[ix])*sin(dec[ij]) + \
                cos(dec[ix])*cos(dec[ij])*cos(fabs(ra[ix]-ra[ij])) );
    }//loop over gals

}

//device code using shared mem
__global__ void CalcSepShared( float* ra, float* dec, int ngals)
{

    //put the ra and dec arrays into shared mem
    __shared__ float raS[128];
    __shared__ float decS[128];

    for(int i=1;i<ngals;i++){
        raS[i] = ra[i];
        decS[i] = dec[i];
    }


    //does all the i's simultaneously - one for each thread 
    int ix = blockDim.x * blockIdx.x + threadIdx.x;

    float sep=0;
    // Do 1 ``column"
    for(int ij=ix+1;ij<ngals;ij++)
    {
        sep = acos( sin(decS[ix])*sin(decS[ij]) + \
                cos(decS[ix])*cos(decS[ij])*cos(fabs(raS[ix]-raS[ij])) );
    }//loop over gals

}




//Host code
int main()
{
    int ngals = 128;


    size_t sizeneededin = ngals * sizeof(float);

    //allocate vectors in host memory
    float* h_ra = (float*)malloc(sizeneededin);
    float* h_dec = (float*)malloc(sizeneededin);
    srand(time(0));

    //initailise input vectors - place galaxies at rando coords between 0 and 1
    for(int i=0;i<ngals;i++)
    {
        h_ra[i] = rand(); 
        h_dec[i] = rand();
    }

    //Calculate separation in CPU and calculate time needed
    clock_t teststart = clock();

    float testsep=0;
    for(int i=0;i<ngals;i++){
        for(int j=i+1;j<ngals;j++){
            testsep = acos( sin(h_dec[i])*sin(h_dec[j]) + cos(h_dec[i])*cos(h_dec[j])*cos(fabs(h_ra[i]-h_ra[j])) );
        }
    }
    clock_t testend = clock();
    float testelapsed = (float)(testend-teststart);
    printf("elapsed time for CPU in ms: %f", testelapsed/CLOCKS_PER_SEC*1000);
    printf("\n");


    //allocate vectors in device memory
    float* d_ra;  float* d_dec; 
    cudaMalloc(&d_ra, sizeneededin);
    cudaMalloc(&d_dec, sizeneededin);

    //copy vectors from host to device memory 
    cudaMemcpy(d_ra, h_ra, sizeneededin, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_dec, h_dec, sizeneededin, cudaMemcpyHostToDevice); 

    //invoke kernel
    int threadsPerBlock = (ngals*ngals)/2;
    // Only need 1/2 as many threads
    //  int blocksPerGrid = (ngals/2 + threadsPerBlock -1) / threadsPerBlock; //???????
    //this is a temp thing - to test I want all threads acessing the same memory
    int blocksPerGrid = 1;

    //set up the cuda timer. 
    cudaEvent_t cudastart, cudaend;
    cudaEventCreate(&cudastart); 
    cudaEventCreate(&cudaend);
    //record the start time
    cudaEventRecord(cudastart,0);

    //run the kernel! 
    CalcSep<<<blocksPerGrid, threadsPerBlock>>>(d_ra, d_dec, ngals);

    //record the end time
    cudaEventRecord(cudaend,0);
    cudaEventSynchronize(cudaend);

    //how long did the kernel take? this gives time in ms
    float cudaelapsed=0;
    cudaEventElapsedTime(&cudaelapsed, cudastart, cudaend);
    printf("elapsed time for GPU in ms: %f",cudaelapsed);
    printf("\n");

    //delete memory
    cudaEventDestroy(cudastart);
    cudaEventDestroy(cudaend);


    //////////////////////////////////////////////////
    //now, do the same thing but with shared  memory
    /////////////////////////////////////////////////


    //set up the cuda timer. 
    cudaEvent_t cudastart2, cudaend2;
    cudaEventCreate(&cudastart2); 
    cudaEventCreate(&cudaend2);
    //record the start time
    cudaEventRecord(cudastart2,0);

    //run the kernel! 
    CalcSepShared<<<blocksPerGrid, threadsPerBlock>>>(d_ra, d_dec, ngals);

    //record the end time
    cudaEventRecord(cudaend2,0);
    cudaEventSynchronize(cudaend2);

    //how long did the kernel take? this gives time in ms
    float cudaelapsed2=0;
    cudaEventElapsedTime(&cudaelapsed2,cudastart2, cudaend2);
    printf("elapsed time for GPU using shared memory in ms: %f",cudaelapsed2);
    printf("\n");

    //delete memory
    cudaEventDestroy(cudastart2);
    cudaEventDestroy(cudaend2);



    //free device memory
    cudaFree(d_ra); cudaFree(d_dec);
    //free host memory
    free(h_ra); free(h_dec); 

printf("size of 128 floats: %f \n", float(128*sizeof(float)));


}
