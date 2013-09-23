import numpy as np
import scipy.stats as stats

lo = [0.0,0.0]
hi = [1.0,1.0]

mux = [.3, 0.6, 0.3, 0.6]
muy = [.3, 0.3, 0.6, 0.6]
sig = 0.05

nspots = 5

npts = 250

xpts = np.array([])
ypts = np.array([])
for mx,my in zip(mux,muy):

    x = np.random.normal(mx,sig,npts)
    xpts = np.append(xpts,x)

    y = np.random.normal(my,sig,npts)
    ypts = np.append(ypts,y)

zipped = zip(xpts,ypts)
np.savetxt('test.out',zipped,fmt='%f %f')



