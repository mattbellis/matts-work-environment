#include <stdio.h>

int main()
{
  int dimx = 16;
  int num_bytes = dimx*sizeof(int);
  int *d_a=0, *h_a=0; // device and host pointers

  h_a = (int*)malloc(num_bytes); //allocate mem on CPU side
  cudaMalloc( (void**)&d_a, num_bytes); //allocate mem on GPU side
  
  // did the mem allocation work? 
  if(0==h_a || 0==d_a)
    {
      printf("couln't allocate memory\n");
      return 1;
    }

  cudaMemset( d_a, 0, num_bytes); //set the GPU memory to 0
  cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost); //cp from GPU to CPU the pointer values

  for(int i=0;i<dimx;i++) 
    printf("%d",h_a[i]);
  printf("\n");

  free(h_a); //free the CPU mem
  cudaFree(d_a); //free th eGPU mem

  return 0;
}
