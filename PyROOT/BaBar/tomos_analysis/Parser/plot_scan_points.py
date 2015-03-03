#!/usr/bin/env python

import sys

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

infile = open(sys.argv[1])


xpts = []
ypts = []
for line in infile:

    vals = line.split()

    print vals

    if len(vals)>=4 and vals[3] is not "4":

        xpts.append(float(vals[1]))
        #ypts.append(float(vals[2]))
        ypts.append(stats.chisqprob(float(vals[2]),22))


#print xpts
#print ypts
print len(xpts)
print len(ypts)
plt.plot(xpts,ypts,"o")
plt.show()

   


