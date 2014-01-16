import matplotlib.pylab as plt
import numpy as np

import sys

from scipy.optimize import curve_fit,leastsq

expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]
errfunc = lambda p, x, y, err: (y - expfunc(p, x)) / err

expfunc1 = lambda p, x: p[1]*x + p[0]
errfunc1 = lambda p, x, y, err: (y - expfunc1(p, x)) / err

################################################################################
# Fit blue data
################################################################################
infile = open('fig30_blue.dat','r')

content = np.array(infile.read().split()).astype('float')
npts = len(content)/2
# Get time
index = np.arange(0,npts*2,2)
xpts = content[index]
ypts = content[index+1]

yerr = np.sqrt(ypts)

plt.figure()
plt.errorbar(xpts,ypts,yerr=yerr,xerr=0.001,fmt='o')

pinit = [1.0,1.0,1.0]
out = leastsq(errfunc, pinit, args=(xpts[0:20],ypts[0:20],yerr[0:20]), full_output=1)
xp = np.linspace(min(xpts),max(xpts[0:20]),1000)
z = out[0]
zcov = out[1]
print "z: ",z
print "zcov: ",zcov
yp = expfunc(z,xp)
plt.plot(xp,yp,'-',color='r')

################################################################################
# Fit blue data
################################################################################
infile = open('fig30_green.dat','r')

content = np.array(infile.read().split()).astype('float')
npts = len(content)/2
# Get time
index = np.arange(0,npts*2,2)
xpts = content[index]
ypts = content[index+1]

yerr = np.sqrt(ypts)

plt.figure()
plt.errorbar(xpts,ypts,yerr=yerr,xerr=0.001,fmt='o')

pinit = [1.0,1.0,1.0]
out = leastsq(errfunc, pinit, args=(xpts[0:19],ypts[0:19],yerr[0:19]), full_output=1)
xp = np.linspace(min(xpts[0:19]),max(xpts[0:19]),1000)
z = out[0]
zcov = out[1]
print "z: ",z
print "zcov: ",zcov
yp = expfunc(z,xp)
plt.plot(xp,yp,'-',color='r')



plt.show()

plt.show()
