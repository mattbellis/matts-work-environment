from numba import autojit, prange
import numpy as np

import time


@autojit
def parallel_sum(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum += A[i]

    return sum

a = np.random.random((10000,20000))
print(a.shape[0])

t0 = time.time()
tot = parallel_sum(a)
#print(tot)
print(np.sum(tot))
print("Time to run: %f sec" % (time.time() - t0))

t0 = time.time()
#a = a.transpose()
tot = np.zeros(a.shape[1])
for i in range(a.shape[1]):
    #tot[i] = a[:,i].sum()
    tot[i] = np.sum(a[:,i])
#print(tot)
print(np.sum(tot))
print("Time to run: %f sec" % (time.time() - t0))
