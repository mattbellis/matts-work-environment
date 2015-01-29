"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


################################################################################
################################################################################
# Make a figure on which to plot stuff.
fig1 = plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
#
# Usage is XYZ: X=how many rows to divide.
#               Y=how many columns to divide.
#               Z=which plot to plot based on the first being '1'.
# So '111' is just one plot on the main figure.
################################################################################
subplots = []
for i in range(1,5):
  division = 220 + i
  subplots.append(fig1.add_subplot(division))

################################################################################
################################################################################

#plt.show()

#"""
# For working on the command line
# filename = sys.argv[1]
# For now when we run in IDLE
filename = "exercise_2_small.txt"
infile = open(filename)

E = []
x = []
y = []
z = []
mass = []

energy = [[], [], [], []]

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

  mass.append(sqrt((E[2]+E[3]+E[4])**2 -  (x[2]+x[3]+x[4])**2 -
                   (y[2]+y[3]+y[4])**2 -  (z[2]+z[3]+z[4])**2 ))
  energy[0].append(E[1])
  energy[1].append(E[2])
  energy[2].append(E[3])
  energy[3].append(E[4])



# the histogram of the data
n, bins, patches = subplots[0].hist(mass,   100, range=(1.78,1.96), normed=0, facecolor='green', alpha=0.75, histtype='stepfilled')

#n, bins, patches = subplots[1].hist(energy[0], 100, range=(0,5), normed=0, facecolor='red', alpha=0.45, histtype='stepfilled')
n, bins, patches = subplots[1].hist(energy[1], 100, range=(0,5), normed=0, facecolor='yellow', alpha=0.45, histtype='stepfilled', label='K')
n, bins, patches = subplots[1].hist(energy[2], 100, range=(0,5), normed=0, facecolor='cyan', alpha=0.45, histtype='stepfilled', label='pi0')
n, bins, patches = subplots[1].hist(energy[3], 100, range=(0,5), normed=0, facecolor='green', alpha=0.45, histtype='stepfilled', label='pi1')

#subplots[1]()
n, bins, patches = plt.hist(energy[0], 100, range=(0,5), normed=0, facecolor='red', alpha=0.45, histtype='stepfilled', label='piB')

#subplots[1].set_ylim(0,17000)

subplots[0].set_xlabel('Mass')
subplots[0].set_ylabel('# events')
#leg = plt.legend(('piB'))

#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#subplots[0].set_xlim(1.78, 1.96)
#ax.set_ylim(0, 0.03)
#ax.grid(True)

plt.show()
#"""
