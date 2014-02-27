import numpy as np

import time


start = time.time()
x = np.zeros((1000000, 10, 10))
print "zeros: %f" % (time.time()-start)
#print x

start = time.time()
x = np.random.random((1000000, 10, 10))
print "random.random: %f" % (time.time()-start)
#print x

# Get all the 1th elements
print len(x[:,1])

# Get all the zeroth elements of the 1th elements
print len(x[:,1,0])
