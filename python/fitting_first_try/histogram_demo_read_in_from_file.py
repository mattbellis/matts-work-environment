"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

filename = sys.argv[1]
infile = open(filename)

E = []
x = []
y = []
z = []
mass = []

xpts = []
ypts = []

count = 0
for line in infile:

  if count%10000 == 0:
    print count
  count += 1

  del E[:]
  del x[:]
  del y[:]
  del z[:]
  vars = line.split()
  nvars = len(vars)
  nparts = int(nvars/5)
  for i in range(0,nparts):
    E.append(float(vars[1 + 5*i]))
    x.append(float(vars[2 + 5*i]))
    y.append(float(vars[3 + 5*i]))
    z.append(float(vars[4 + 5*i]))
    #mass.append(sqrt(E[i]**2 - x[i]**2 - y[i]**2 - z[i]**2))

    if i==1:
      xpts.append(x[i])
      ypts.append(y[i])

  mass.append(sqrt((E[2]+E[3]+E[4])**2 -  (x[2]+x[3]+x[4])**2 -  (y[2]+y[3]+y[4])**2 -  (z[2]+z[3]+z[4])**2 ))


fig = plt.figure()
ax = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(121)

# the histogram of the data
n, bins, patches = ax.hist(mass, 500, normed=0, facecolor='green', alpha=0.75, histtype='stepfilled')

print len(xpts)
print len(ypts)
ax2.hexbin(xpts, ypts)

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
#y = mlab.normpdf( bincenters, mu, sigma)
#l = ax.plot(bincenters, y, 'r--', linewidth=1)

ax.set_xlabel('Mass')
ax.set_ylabel('# events')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.set_xlim(1.78, 1.96)
#ax.set_ylim(0, 0.03)
#ax.grid(True)

plt.show()
