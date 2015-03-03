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

from pylab import errorbar

#from math import *

def fitfunc(p, x):
  ret = p[0] + p[1]*x
  return ret


def errfunc(p, x, y, er):
  ret = (fitfunc(p, x)-y)/(er*1.0)

  return ret



mpn2 = [1.88907,5.66721,9.44536,13.2235,17.0016,20.7798,24.5579]

sigL = [3.03212e-03,2.97864e-03,2.94196e-03,3.15229e-03,2.92145e-03,2.88793e-03,3.06631e-03]
sigLErr = [2.26176e-05,2.22000e-05,3.82006e-05,1.39427e-04,4.68567e-05,2.42062e-05,2.19026e-05]

sigR = [2.41147e-03,2.42517e-03,2.42845e-03,2.34186e-03,2.37614e-03,2.40756e-03,2.63241e-03]
sigRErr = [1.71844e-05,1.70601e-05,2.92415e-05,1.04646e-04,3.56305e-05,1.84822e-05,1.71735e-05]

#	mpn2 = [1.88907,5.66721,9.44536,13.2235,17.0016,20.7798,24.5579]
#	
#	sigL = [3.16652e-03,3.12990e-03,3.09206e-03,3.22277e-03,3.07291e-03,3.05197e-03,3.23667e-03]
#	sigLErr = [1.98810e-05,1.94934e-05,3.37253e-05,1.27210e-04,4.10472e-05,2.10766e-05,2.01318e-05]
#	
#	sigR = [2.37219e-03,2.37996e-03,2.38075e-03,2.31643e-03,2.33120e-03,2.35608e-03,2.58860e-03]
#	sigRErr = [1.63878e-05,1.63320e-05,2.83876e-05,1.04438e-04,3.45256e-05,1.77660e-05,1.68679e-05]

pydatasetx = mpn2
pydatasety = sigL
pydatasetyErr = sigLErr

fig = plt.figure()
ax = fig.add_subplot(111)

xpts = array(pydatasetx)
ypts = array(pydatasety)
#xpts = array([0,1,2,3,4,5,6])
#ypts = array([0,1,2,3,4,5,6])
erpts = array(pydatasetyErr)

p0 = [0.001, 0.001 ]
p1, success = optimize.leastsq(errfunc, p0[:], args=(xpts, ypts, erpts))

print success
print p1

testfitfunc = lambda x: fitfunc(p1, x)
val, err = integrate.quadrature(testfitfunc, 40, 160)
print err

masses = linspace(xpts.min(),xpts.max(),100)
m = ax.plot(xpts, ypts, "ro", masses , fitfunc(p1, masses), "r-") # Plot of the data and the fit
errorbar(xpts, ypts, yerr=erpts, fmt='ro')

ax.set_xlabel('mpn2')
ax.set_ylabel('Sigma')
ax.grid(True)

#
plt.show()
#"""
