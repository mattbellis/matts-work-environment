#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include"magma.h"
#include"magma_lapack.h"

float rand_normal(){

    // Use the Box-Muller method
    float U = rand() / (double)RAND_MAX;
    float V = rand() / (double)RAND_MAX;

    float X = sqrt(-2*log(U))*cos(2*3.14159*V);
    float Y = sqrt(-2*log(U))*sin(2*3.14159*V);

    return X;
}



int main ( int argc , char ** argv ) {
    magma_init (); // initialize Magma
    magma_int_t n=128 , n2=n*n;
    //magma_int_t n=256 , n2=n*n;
    //magma_int_t n=512 , n2=n*n;
    //magma_int_t n=1024 , n2=n*n;
    //magma_int_t n=2048 , n2=n*n;
    //magma_int_t n=4096 , n2=n*n;
    //magma_int_t n=8192 , n2=n*n;
    //magma_int_t n=16384 , n2=n*n;

    float *a, *r; // a, r - nxn matrices on the host
    float *d_r; // nxn matrix on the device
    float * h_work ; // workspace

    magma_int_t lwork ; // h_work size
    magma_int_t * iwork ; // workspace
    magma_int_t liwork ; // iwork size

    float *w1 , *w2; // w1 ,w2 - vectors of eigenvalues
    float error , work [1]; // used in difference computations

    magma_int_t ione = 1 , i, j, info ;

    float mione = -1.0f;

    magma_int_t incr = 1 , inci = 1;
    magma_smalloc_cpu (&w1 ,n); // host memory for real
    magma_smalloc_cpu (&w2 ,n); // eigenvalues
    magma_smalloc_cpu (&a,n2 ); // host memory for a
    magma_smalloc_cpu (&r,n2 ); // host memory for r
    magma_smalloc (& d_r ,n2 ); // device memory for d_r

    // Query for workspace sizes
    float aux_work [1];

    magma_int_t aux_iwork [1];
    magma_ssyevd_gpu ('V','L',n,d_r ,n,w1 ,r,n, aux_work , -1 ,

            aux_iwork , -1 ,& info );
    lwork = ( magma_int_t ) aux_work [0];

    liwork = aux_iwork [0];

    iwork =( magma_int_t *) malloc ( liwork * sizeof ( magma_int_t ));

    magma_smalloc_cpu (& h_work , lwork ); // memory for workspace

    /*
    ////////////////////////////////////////////////////////////////////////////
    // Testing. Only diagonal.
    ////////////////////////////////////////////////////////////////////////////
    // define a, r // [1 0 0 0 0 ...]
    for(i=0;i<n;i ++){ // [0 2 0 0 0 ...]
        a[i*n+i]=( float )(i +1); // a = [0 0 3 0 0 ...]
        r[i*n+i]=( float )(i +1); // [0 0 0 4 0 ...]
    } // [0 0 0 0 5 ...]
    */

    ///*
    ////////////////////////////////////////////////////////////////////////////
    // Random, symmetric matrix
    ////////////////////////////////////////////////////////////////////////////
    //int seed = (int) time(NULL);
    int seed = 10;
    srand(seed);

    // define a, r // 
    for(i=0;i<n;i ++){ // 
        for(j=i;j<n;j ++){ // 
            float num = rand_normal();
            a[i*n+j]=num;
            a[j*n+i]=num;

            r[i*n+j]=num;
            r[j*n+i]=num;
        } 
    } 
    //*/

    printf (" upper left corner of a:\n"); // .............
    magma_sprint (5 ,5 ,a,n); // print part of a
    //magma_sprint (n ,n ,a,n); // print part of a
    magma_ssetmatrix ( n, n, a, n, d_r , n); // copy a -> d_r
    ////////////////////////////////////////////////////////////////////////////
    // compute the eigenvalues and eigenvectors for a symmetric ,
    // real nxn matrix ; Magma version
    ////////////////////////////////////////////////////////////////////////////
    clock_t start = clock(), diff;
    magma_ssyevd_gpu(MagmaVec,MagmaLower,n,d_r,n,w1,r,n, h_work,lwork,iwork,liwork,&info);
    //magma_ssyevd_gpu(MagmaNoVec,MagmaLower,n,d_r,n,w1,r,n, h_work,lwork,iwork,liwork,&info);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds", msec/1000, msec%1000);
    ////////////////////////////////////////////////////////////////////////////
    printf (" first 5 eigenvalues of a:\n");
    for(j=0;j<5;j++)
        printf ("%f\n",w1[j]); // print first eigenvalues
    /*
    printf (" left upper corner of the matrix of eigenvectors :\n");
    magma_sgetmatrix ( n, n, d_r , n, r, n ); // copy d_r -> r
    magma_sprint (5 ,5 ,r,n); // part of the matrix of eigenvectors
    //magma_sprint (n ,n ,r,n); // part of the matrix of eigenvectors
    // Lapack version
    //lapackf77_ssyevd ("V","L" ,&n,a ,&n,w2 ,h_work ,& lwork ,iwork , &liwork ,& info );
    // difference in eigenvalues
    //blasf77_saxpy ( &n, &mione , w1 , &incr , w2 , & incr );
    //error = lapackf77_slange ( "M", &n, &ione , w2 , &n, work );
    //printf (" difference in eigenvalues : %e\n",error );
    
    ///////// MY TEST ///////////////
    printf("\n\n");
    for(i=0;i<n;i++)
    {
        float tot = 0;
        //float val = r[1*n+i]; // Column of eigenvectors
        float val = r[i*n]; // Column of eigenvectors
        for(j=0;j<n;j++)
        {
            printf ("%6.3f ",a[j*n+i]); // print matrix
            tot += r[1*n+j]*a[j*n+i];
            if (j==n-1)
                printf ("\t%6.3f",r[1*n+i]); // print first eigenvectors
            //printf ("%f\n",w1[j]); // print first eigenvalues
        }
        //printf ("\t%6.3f",val); // print first eigenvectors
        printf ("\t\t%6.3f",tot); // print first eigenvectors
        printf ("\t\t%6.3f",tot/w1[1]); // print first eigenvectors
        printf("\n");
    }
    */
    free (w1 ); // free host memory
    free (w2 ); // free host memory
    free (a); // free host memory
    free (r); // free host memory
    free ( h_work ); // free host memory
    magma_free (d_r ); // free device memory
    magma_finalize (); // finalize Magma
    return EXIT_SUCCESS ;
}
// upper left corner of a:
//[
// 1.0000 0. 0. 0. 0.
// 0. 2.0000 0. 0. 0.
// 0. 0. 3.0000 0. 0.
// 0. 0. 0. 4.0000 0.
// 0. 0. 0. 0. 5.0000
// ];
// first 5 eigenvalues of a:
// 1.000000
// 2.000000
// 3.000000
// 4.000000
// 5.000000
// left upper corner of the matrix of eigenvectors :
// [
// 1.0000 0. 0. 0. 0.
// 0. 1.0000 0. 0. 0.
// 0. 0. 1.0000 0. 0.
// 0. 0. 0. 1.0000 0.
// 0. 0. 0. 0. 1.0000
// ];
// difference in eigenvalues : 0.000000 e+00


