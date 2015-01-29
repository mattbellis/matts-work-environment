import numpy as np
from time import time

nentries = 10000000

x = np.random.random(nentries)
xl = x.tolist()

# Array
st = time()
for n in x:
    a = n
diff = time() - st
print "time for array: %f" % (diff)


# Array (nditer)
st = time()
for n in np.nditer(x):
    a = n
diff = time() - st
print "time for nditer: %f" % (diff)

# Array index
st = time()
for n in xrange(len(x)):
    a = x[n]
diff = time() - st
print "time for nditer: %f" % (diff)


# List
st = time()
for n in xl:
    a = n
diff = time() - st
print "time for list:  %f" % (diff)


