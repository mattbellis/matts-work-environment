######### DOESN'T WORK RIGHT NOW! ###################
from numpy import random,zeros
from numba import double
from numba.decorators import jit as jit

import time

####################################################################
def mult2arr_np(x,y):

    result = x*y

    return result
################################################################################

####################################################################
def mult2arr(x,y):

    ix = len(x)
    jy = len(y)
    result = zeros(ix)
    for i in range(ix):
        result[i] = x[i]*y[i]

    return result
################################################################################

cmult2arr = jit(restype=double[:,:], argtypes=[double[:,:],double[:,:]])(mult2arr)

x = random.random(100000)
y = random.random(100000)

################################################################################

start = time.time()
res = mult2arr(x,y)
duration = time.time() - start
print "Result from python is %s in %s (msec)" % (res, duration*1000)

################################################################################

start = time.time()
res = cmult2arr(x,y)
duration2 = time.time() - start
print "Result from compiled is %s in %s (msec)" % (res, duration2*1000)

print "Speed up is %s" % (duration / duration2)

################################################################################

start = time.time()
res = mult2arr_np(x,y)
duration3 = time.time() - start
print "Result from numpy is %s in %s (msec)" % (res, duration3*1000)

################################################################################
