from numba import autojit, prange

@autojit
def parallel_sum(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum += A[i]

    return sum

import numpy as np

a = np.random.random((100,100))

print(parallel_sum(a))
