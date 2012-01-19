#!/usr/bin/env python
"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import numpy as np
#from numpy import *
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import scipy
import pylab
from scipy import *
from scipy import optimize
from scipy import integrate
#from math import *

mu, sigma = 100, 15
data_points = mu + sigma * np.random.randn(10000)

fig = plt.figure()
ax = fig.add_subplot(111)

# the histogram of the data
n, bins, patches = ax.hist(data_points, 50, normed=1, facecolor='green', alpha=0.75)
bincenters = 0.5*(bins[1:]+bins[:-1])

xpts = linspace(0, 10, len(bincenters))
ypts = linspace(0, 10, len(n))
for i in range(0,len(xpts)):
  xpts[i] = bincenters[i]
  ypts[i] = n[i]

p0 = [0.02, 100, 20]
fitfunc = lambda p, x: (p[0]/p[2])*exp(-((x - p[1])**2)/(2.0*p[2]*p[2]))
#fitfunc = lambda p, x: p[0]*exp(-((x - p[1])**2)/(2*p[2]*p[2]))
errfunc = lambda p, x, y: fitfunc(p, x)-y
p1, success = optimize.leastsq(errfunc, p0[:], args=(xpts, ypts))

print success
print p1

#val, err = integrate.quadrature(fitfunc(p1, x), 50, 150)
#print "nevents: %f +/- %f" % (val, err)

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
# add a 'best fit' line for the normal PDF
#"""
y = mlab.normpdf( bincenters, mu, sigma)
l = ax.plot(bincenters, y, 'r--', linewidth=1)

x = np.arange(40.0, 150.0, 0.01)
p = p1
plotfunc = fitfunc(p1, x)
m = ax.plot(x, plotfunc, 'black', linewidth=5)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.set_xlim(40, 160)
ax.set_ylim(0, 0.03)
ax.grid(True)

plt.show()
#"""
