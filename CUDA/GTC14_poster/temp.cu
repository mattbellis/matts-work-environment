__global__ void kernel (int *a)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = idx;
}

int main()
{
    .
    .
    kernel<<<grid,block>>>(d_a);
    .
    .
}
