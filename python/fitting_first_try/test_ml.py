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

p0 = [0.02, 100, 20]
fitfunc = lambda p, x: (p[0]/p[2])*exp(-((x - p[1])**2)/(2*p[2]*p[2]))
#errfunc = lambda p, x, y: ((fitfunc(p, x)-y)**2).sum()
errfunc = lambda p, x, y: (-log(fitfunc(p, x)) ).sum()

xpts = linspace(0,10,11)
ypts = linspace(0,10,11)

print errfunc(p0, xpts, ypts)

print len(xpts)
print "Starting..."
#p1 = fmin(errfunc, p0, args=(xpts, ypts), maxiter=10000, maxfun=10000)
p1 = fmin(errfunc, p0, args=(xpts, xpts), maxiter=10000, maxfun=10000)
print "Ending..."

print p1

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
# add a 'best fit' line for the normal PDF
#"""
#"""
