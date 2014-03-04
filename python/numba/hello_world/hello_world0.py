######### Working! ###################
import numpy as np
from numba import double
from numba.decorators import jit as jit
from numba.decorators import autojit 

import time

####################################################################
def mult2arr_np(x,y):

    result = x*y

    return result
################################################################################

####################################################################
def mult2arr(x,y):

    ix = len(x)
    result = np.zeros(ix)
    for i in range(ix):
        result[i] = x[i]*y[i]

    return result
################################################################################

cmult2arr = jit(restype=double[:], argtypes=[double[:],double[:]])(mult2arr)
#cmult2arr = autojit(mult2arr)

x = np.random.random(1000000)
y = np.random.random(1000000)

################################################################################

start = time.time()
res = mult2arr(x,y)
duration = time.time() - start
print "Result from python is %s\nin %s (msec)" % (res[0:3], duration*1000)

################################################################################

start = time.time()
res = cmult2arr(x,y)
duration2 = time.time() - start
print "Result from compiled is %s\nin %s (msec)" % (res[0:3], duration2*1000)

#print "Speed up is %s" % (duration / duration2)

################################################################################

start = time.time()
res = mult2arr_np(x,y)
duration3 = time.time() - start
print "Result from numpy is %s\nin %s (msec)" % (res[0:3], duration3*1000)

################################################################################
