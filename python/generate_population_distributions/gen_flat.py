import numpy as np

lo = [0.0,0.0]
hi = [1.0,1.0]

npts = 100000

vals = []
for l,h in zip(lo,hi):

    width = h-l
    v = width*np.random.random(npts) + l
    print v
    vals.append(v)

zipped = zip(vals[0],vals[1])
np.savetxt('test.out',zipped,fmt='%f %f')



