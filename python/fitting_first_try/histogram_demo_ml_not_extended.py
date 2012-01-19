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
from scipy.optimize import fmin
#from math import *

nevents = 10000
mu, sigma = 100, 15
data_points = mu + sigma * np.random.randn(nevents)

norm_points = (mu - 4.0*sigma) + 8.0*sigma * np.random.random(nevents*10)

print data_points

fig = plt.figure()
ax = fig.add_subplot(111)

# the histogram of the data
#n, bins, patches = ax.hist(data_points, 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = ax.hist(data_points, 50, facecolor='green', alpha=0.75)
#n0, bins0, patches0 = ax.hist(norm_points, 50, facecolor='blue', alpha=0.15)
bincenters = 0.5*(bins[1:]+bins[:-1])

def pois(p, k):
  mu = p[2]
  ret = -mu + k*log(mu)
  return ret

def fitfunc(p, x):
  ret = (1.0/p[1])*exp(-((x - p[0])**2)/(2.0*p[1]*p[1]))
  return ret

def errfunc(p, x, y):
  norm_func = (fitfunc(p, y)).sum()/len(y)
  ret = (-log(fitfunc(p, x) / norm_func) .sum())
  #print "%f  %f" % (ret, norm_func)
  return ret

#p0 = [100, 15, nevents]
p0 = [90, 10, nevents]

print "Starting..."
p1 = fmin(errfunc, p0, args=(data_points, norm_points), maxiter=10000, maxfun=10000)
print "Ending..."

print p1

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
# add a 'best fit' line for the normal PDF
#"""
#y = mlab.normpdf( bincenters, mu, sigma)
#l = ax.plot(bincenters, y, 'r--', linewidth=1)

x = np.arange(40.0, 200.0, 0.01)

# Get the normalization
binwidth = bincenters[1] - bincenters[0]
print binwidth
norm = sqrt(2*3.14)
print "norm: %f" %(norm)
print "binwidth: %f" %(binwidth)
print "p1[2]: %f" %(p1[2])
scale = (p1[2]*binwidth)/norm

plotfunc = scale*fitfunc(p1, x)
m = ax.plot(x, plotfunc, 'black', linewidth=5)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.set_xlim(40, 160)
#ax.set_ylim(0, 0.03)
ax.grid(True)

plt.show()
#"""
