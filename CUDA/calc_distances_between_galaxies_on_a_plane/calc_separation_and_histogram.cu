#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//// notes

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// GPU code to calculate the bin number.
// This assumes that you have normalized your data that you want to plot to 
// lie between 0.0 and 1.0. 
// Outside this range is classified as overflow or underflow.
///////////////////////////////////////////////////////////////////////////////
__device__ int get_bin_num(float normalized_val, int nbins)
{
    // The data goes in bins number 1 to nbins
    // 0 is the underflow
    // nbins-1 is the overflow
    //
    // Remember that we have underflow (0) and overflow (nbins-1) 
    if (normalized_val>=1.0)
    {
        // If it is greater than or equal to 1.0, put it in the overflow bin
        return nbins-1;
    }
    else if (normalized_val<0.0)
    {
        return 0;
    }
    else if (normalized_val==0.0)
    {
        return 1;
    }

    // Do this calculation only if it fails the other conditionals.
    // I think this buys us a few CPU cycles.
    int ret = (int)(normalized_val*(nbins-2)) + 1;    
    return ret;
}

///////////////////////////////////////////////////////////////////////////////
// GPU code to calculate the separation between two galaxies given the 
// right ascenscion and declanation.
///////////////////////////////////////////////////////////////////////////////
//__global__ void CalcSep(float* raA, float* decA, int ngals, int nthreads, int* hist_array, float hist_lo, float hist_hi, int hist_nbins)
__global__ void CalcSep(float* raA, float* sin_decA, float* cos_decA, int ngals, int nthreads, int* hist_array, float hist_lo, float hist_hi, int hist_nbins)
{
    //does all the i's simultaneously - one for each thread 
    int ix = blockDim.x * blockIdx.x + threadIdx.x;

    // Get normalization term
    float norm = hist_hi-hist_lo;
    float norm_val = 0;
    int bin = 0;
    int hist_array_bin_block = ix*hist_nbins;
    int hist_array_bin = 0;

    float sep=0;
    float sin_dec_ix,sin_dec_ij;
    float cos_dec_ix,cos_dec_ij;
    float ra_ix, ra_ij;
    // Do the ix ``column"
    for(int ij=ix+1;ij<ngals;ij++)
    {

        sin_dec_ix = sin_decA[ix];
        sin_dec_ij = sin_decA[ij];
        cos_dec_ix = cos_decA[ix];
        cos_dec_ij = cos_decA[ij];
        ra_ix = raA[ix];
        ra_ij = raA[ij];

        sep = acos( sin_dec_ix*sin_dec_ij + cos_dec_ix*cos_dec_ij*cos(fabs(ra_ix-ra_ij)) );

        norm_val = (sep-hist_lo)/norm;
        bin = get_bin_num(norm_val,hist_nbins);
        hist_array_bin = hist_array_bin_block + bin;

        // If we passed 0 bins or -x on the command line, don't try
        // to fill the super array.
        if (hist_nbins>2)
        {
            hist_array[hist_array_bin]++;
        }

    }//loop over gals

    // Then the ngals-ix ``column"
    ix = (ngals - 1) - ix;
    for(int ij=ix+1;ij<ngals;ij++)
    {
        sin_dec_ix = sin_decA[ix];
        sin_dec_ij = sin_decA[ij];
        cos_dec_ix = cos_decA[ix];
        cos_dec_ij = cos_decA[ij];
        ra_ix = raA[ix];
        ra_ij = raA[ij];

        sep = acos( sin_dec_ix*sin_dec_ij + cos_dec_ix*cos_dec_ij*cos(fabs(ra_ix-ra_ij)) );
                
        norm_val = (sep-hist_lo)/norm;
        bin = get_bin_num(norm_val,hist_nbins);
        hist_array_bin = hist_array_bin_block + bin;
        
        // If we passed 0 bins or -x on the command line, don't try
        // to fill the super array.
        if (hist_nbins>2)
        {
            hist_array[hist_array_bin]++;
        }
    }//loop over gals
}
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//Host code
int main(int argc, char **argv)
{
    int ngals = 2000;
    int nbins = 100;
    srand(time(0));

    ///////////////////////////////////////////////////////////////////////////
    // Grab the number of galaxies from the command line *if* they have 
    // been specified.
    if (argc>1)
    {
        ngals = atoi(argv[1]);
        if (argc>2)
        {
            nbins = atoi(argv[2]);
        }
    }
    else
    {
        printf("Usage: %s <number of galaxies> <number of histogram bins>\n",\
                argv[0]);
        printf("\nDefault is 1000 galaxies and 100 bins\n\n"); 
    }
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Define histo arrays and memory info and the like
    size_t gal_mem_needed = ngals * sizeof(float);

    // How many threads will there be?
    int nthreads = ngals/2;
    printf("nthreads: %d\n",nthreads);

    float hist_lo = 0.0;
    float hist_hi = 3.5;

    // From here on out, use the number of bins with underflow/overflow added in
    // to the calculation.
    int nbins_with_overflow = nbins + 2;
    int nbins_in_super_hist_array = nthreads*nbins_with_overflow;

    size_t hist_mem_needed = nbins_in_super_hist_array*sizeof(int);

    ///////////////////////////////////////////////////////////////////////////
    //allocate vectors in host memory
    float* h_raA = 0; 
    float* h_decA = 0;

    float* h_sin_decA = 0; 
    float* h_cos_decA = 0;

    int *h_hist_array = 0;
    int *h_hist_array_compressed = 0;

    h_raA = (float*)malloc(gal_mem_needed);
    h_decA = (float*)malloc(gal_mem_needed);

    // Allocate memory for the cos and sin of the right asenscion. This saves
    // us some time rather than recalcuating this over and over on the GPUs.
    h_cos_decA = (float*)malloc(gal_mem_needed);
    h_sin_decA = (float*)malloc(gal_mem_needed);

    h_hist_array = (int*)malloc(hist_mem_needed);
    h_hist_array_compressed = (int*)malloc(nbins_with_overflow*sizeof(int));

    if (0==h_raA || 0==h_sin_decA || 0==h_cos_decA || 0==h_hist_array || 0==h_hist_array_compressed)
    {
        printf("Couldn't allocate memory on host....\n");
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    //allocate vectors in device memory
    float* d_raA=0;
    //float* d_decA=0;

    float* d_sin_decA=0;
    float* d_cos_decA=0;

    int *d_hist_array;

    cudaMalloc(&d_raA, gal_mem_needed);
    //cudaMalloc(&d_decA, gal_mem_needed);

    cudaMalloc(&d_cos_decA, gal_mem_needed);
    cudaMalloc(&d_sin_decA, gal_mem_needed);

    cudaMalloc(&d_hist_array, hist_mem_needed);

    if (0==d_raA || 0==d_cos_decA || 0==d_sin_decA || 0==d_hist_array)
    {
        printf("Couldn't allocate memory on device....\n");
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Initialise input vectors.
    // Place galaxies at random coords between 0 and 1.
    ///////////////////////////////////////////////////////////////////////////
    for(int i=0;i<ngals;i++)
    {

        h_raA[i] = rand(); 
        h_decA[i] = rand();

        h_cos_decA[i] = cos(h_decA[i]);
        h_sin_decA[i] = sin(h_decA[i]);

    }

    ///////////////////////////////////////////////////////////////////////////
    // Uncomment this section if you would also like to do the calculation on
    // the CPU.
    ///////////////////////////////////////////////////////////////////////////
    /*
    //calculate separation in CPU and calculate time needed
    clock_t teststart = clock();

    float testsep=0;
    for(int i=0;i<ngals;i++){
        for(int j=i+1;j<ngals;j++){
            testsep = acos( sin(h_decA[i])*sin(h_decA[j]) + \
            cos(h_decA[i])*cos(h_decA[j])*cos(fabs(h_raA[i]-h_raA[j])) );
        }
    }
    clock_t testend = clock();
    float testelapsed = (float)(testend-teststart);
    printf("elapsed time for CPU in ms: %f", testelapsed/CLOCKS_PER_SEC*1000);
    printf("\n");
    */
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Zero out the super-array that will hold the histogram entries
    // for each thread.
    printf("nbins_in_super_hist_array: %d\n",nbins_in_super_hist_array);
    for (int i=0;i<nbins_in_super_hist_array;i++)
    {
        h_hist_array[i]=0.0;
    }
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Copy vectors from host to device memory.
    cudaMemcpy(d_raA,  h_raA,  gal_mem_needed, cudaMemcpyHostToDevice); 
    //cudaMemcpy(d_decA, h_decA, gal_mem_needed, cudaMemcpyHostToDevice); 

    cudaMemcpy(d_sin_decA, h_sin_decA, gal_mem_needed, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_cos_decA, h_cos_decA, gal_mem_needed, cudaMemcpyHostToDevice); 

    cudaMemcpy(d_hist_array, h_hist_array, hist_mem_needed, cudaMemcpyHostToDevice); 

    ///////////////////////////////////////////////////////////////////////////
    // Calculate our thread/grid/block sizes.
    int threadsPerBlock = 256;

    // IS THIS CALCULATION BEING DONE PROPERLY? OPTIMALLY?????
    int blocksPerGrid = (nthreads + threadsPerBlock -1) / threadsPerBlock; //???????

    // Set up the cuda timer. 
    // Ccan't use simple CPU timer since that would only time the kernel launch overhead. 
    // Need to make sure all threads have finished before stop the timer 
    // so can synchronise threads before and after kernel launch if using cpu timer? 
    // I didn't get sensible results when I've tried that though. 

    cudaEvent_t cudastart, cudaend;
    cudaEventCreate(&cudastart); 
    cudaEventCreate(&cudaend);

    //record the start time
    cudaEventRecord(cudastart,0);

    ///////////////////////////////////////////////////////////////////////////
    // Run the kernel! 
    //CalcSep<<<blocksPerGrid, threadsPerBlock>>>(d_raA, d_decA, ngals, nthreads, d_hist_array, hist_lo, hist_hi, nbins_with_overflow);
    CalcSep<<<blocksPerGrid, threadsPerBlock>>>(d_raA, d_sin_decA, d_cos_decA, ngals, nthreads, d_hist_array, hist_lo, hist_hi, nbins_with_overflow);

    // Copy the info back off the GPU to the host.
    cudaMemcpy(h_hist_array, d_hist_array, hist_mem_needed, cudaMemcpyDeviceToHost); 

    ///////////////////////////////////////////////////////////////////////////
    // Record the end time
    cudaEventRecord(cudaend,0);
    cudaEventSynchronize(cudaend);

    ///////////////////////////////////////////////////////////////////////////
    // How long did the kernel take? this gives time in ms
    float cudaelapsed=0;
    cudaEventElapsedTime(&cudaelapsed, cudastart, cudaend);
    printf("elapsed time for GPU in ms: %f\n",cudaelapsed);

    ///////////////////////////////////////////////////////////////////////////
    // Collapse the super histogram array to a simple histogram array and write
    // it out to histogram_array.txt
    int sum = 0;
    int master_bin = 0;
    for (int i=0;i<nbins_in_super_hist_array;i++)
    {
        //printf("%d ",h_hist_array[i]);
        sum += h_hist_array[i];

        master_bin = i%nbins_with_overflow;

        //printf("%d\n",master_bin);
        h_hist_array_compressed[master_bin] += h_hist_array[i];
    }
    printf("\ntotal: %d\n",sum);

    FILE *outfile; 
    outfile = fopen("histogram_array.txt","w+"); /* write to file (add text to 
                                                    a file or create a file if it does not exist.*/ 
    // Print out the compressed array
    fprintf(outfile,"%f %f\n",hist_lo,hist_hi);
    for (int i=0;i<nbins_with_overflow;i++)
    {
        fprintf(outfile,"%d ",h_hist_array_compressed[i]);
    }
    fprintf(outfile,"\n");
    fclose(outfile); /*done!*/ 

    ///////////////////////////////////////////////////////////////////////////
    // Free up the device memory.
    cudaEventDestroy(cudastart);
    cudaEventDestroy(cudaend);

    cudaFree(d_raA); 
    //cudaFree(d_decA);
    cudaFree(d_sin_decA);
    cudaFree(d_cos_decA);
    cudaFree(d_hist_array);

    // Free up the host memory.
    free(h_raA); 
    free(h_decA); 
    free(h_sin_decA); 
    free(h_cos_decA); 
    free(h_hist_array);
    free(h_hist_array_compressed);


}
