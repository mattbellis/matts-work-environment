# include <stdio.h>
# include <cuda.h>
# include "magma.h"
# include "magma_lapack.h"

int main ( int argc , char ** argv ) {
    magma_init ();
    // initialize Magma
    magma_timestr_t start , end;
    float gpu_time , cpu_time ;
    magma_int_t n=8192 , n2=n*n;
    float *a, *r;
    // a, r - nxn matrices on the host
    float *d_r;
    // nxn matrix on the device
    float * h_work ;
    // workspace
    magma_int_t lwork ; // h_work size
    magma_int_t * iwork ; // workspace
    magma_int_t liwork ; // iwork size
    float *w1 , *w2; // w1 ,w2 - vectors of eigenvalues
    float error , work [1]; // used in difference computations
    magma_int_t ione = 1 , i, j, info ;
    float mione = -1.0f;
    magma_int_t incr = 1 , inci = 1;
    magma_int_t ISEED [4] = {0 ,0 ,0 ,1}; // seed

    magma_smalloc_cpu (&w1 ,n);
    // host memory for real
   
    magma_smalloc_cpu (&w2 ,n); // eigenvalues
    magma_smalloc_cpu (&a,n2 ); // host memory for a
    magma_smalloc_cpu (&r,n2 ); // host memory for r
    magma_smalloc (& d_r ,n2 ); // device memory for d_r
    // Query for workspace sizes
    float aux_work [1];
    magma_int_t aux_iwork [1];
    magma_ssyevd_gpu ('V','L',n,d_r ,n,w1 ,r,n, aux_work , -1 ,aux_iwork , -1 ,& info );
    lwork = ( magma_int_t ) aux_work [0];
    liwork = aux_iwork [0];
    iwork =( magma_int_t *) malloc ( liwork * sizeof ( magma_int_t ));
    magma_smalloc_cpu (& h_work , lwork );
    // memory for workspace// Random matrix a, copy a -> rlapackf77_slarnv (& ione ,ISEED ,&n2 ,a);
    lapackf77_slacpy ( MagmaUpperLowerStr ,&n ,&n,a ,&n,r ,&n);
    magma_ssetmatrix ( n, n, a, n, d_r , n);
    // copy a -> d_r// compute the eigenvalues and eigenvectors for a symmetric ,// real nxn matrix ;
    //Magma version
    start = get_current_time ();
    magma_ssyevd_gpu(MagmaVec,MagmaLower,n,d_r,n,w1,r,n,h_work,lwork,iwork,liwork,&info);
    end = get_current_time ();
    gpu_time = GetTimerValue (start ,end) / 1e3;
    printf (" ssyevd gpu time : %7.5 f sec .\n",gpu_time );
    // Mag . time// Lapack versionstart = get_current_time ();
    lapackf77_ssyevd ("V","L" ,&n,a ,&n,w2 ,h_work ,& lwork ,iwork ,&liwork ,& info );
    end = get_current_time ();
    cpu_time = GetTimerValue (start ,end) / 1e3;
    printf (" ssyevd cpu time : %7.5 f sec .\n",cpu_time );
    // Lapack// difference in eigenvalues // timeblasf77_saxpy ( &n, &mione , w1 , &incr , w2 , & incr );
    error = lapackf77_slange ( "M", &n, &ione , w2 , &n, work );
    printf (" difference in eigenvalues : %e\n",error );
    free (w1 ); // free host memory
    free (w2 ); // free host memory
    free (a); // free host memory
    free (r); // free host memory
    free ( h_work ); // free host memory
    magma_free (d_r ); // free device memory
    magma_finalize (); // finalize Magma
    return EXIT_SUCCESS ;
}
    //ssyevd gpu time : 19.50048 sec . 
    // 1 GPU// ssyevd cpu time : 19.86725 sec . // 2 CPUs// difference in eigenvalues : 1.358032e -04
