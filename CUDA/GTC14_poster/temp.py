from numba import cuda

@numba.cuda.jit("void(float32[:])")
def kernel(arr_a):
    idx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    arr_a[idx] = idx
.
.
.
kernel[block_ct,thread_ct](a)
