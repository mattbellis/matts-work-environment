#!/usr/bin/env python

from math import *
import numpy as np
import sys
import random as rnd 

#x = [0.000,0.446,0.040,0.307,0.311,0.305,0.085,0.072,0.008,0.364,0.002,0.134,0.025]

################################################################################
# Read in the values from the file.
################################################################################
file = open(sys.argv[1])
pts = []
for line in file:
    vals = line.split()
    pts.append(float(vals[2]))

npts = len(pts)

x = []
if len(sys.argv)==2:
    x = pts
else:
   n_subsamples = int(sys.argv[2])
   do_replacement = False
   if(sys.argv[2] == 'replacement'):
       do_replacement = True
   if not do_replacement:
       # Without replacement
       x = rnd.sample(pts,n_subsamples)
   else:
       # With replacement
       for i in xrange(n_subsamples):
           index = rnd.randint(0,npts-1)
           x.append(pts[index])

#ncounts = 1000

npts = len(x)

print npts 

print np.mean(x)
print np.std(x)

#exit()


################################################################################
# Try the jackknife technique
################################################################################

ncounts = 1

print "Starting jk calculations....."

sub_means = []

max = npts+1

for k in xrange(max):

    y = [] # Stores sub samples.
    n = 0.0
    for j,i in enumerate(x):

        if (k<npts and j!=k) or (k==npts):
            y.append(i)
            n+=1
        
    sub_counts = len(y)
    eff = np.mean(y)

    if (k<npts):
        sub_means.append(eff)
    #print eff

    #err = sqrt((eff*(1.0-eff))/ncounts)
    #err = np.std(sub_means)
    err = np.std(y)
    print "%02d %f +/- %f" % (k, eff,err)

print sub_means
print "%f %f" % (np.mean(sub_means), np.std(sub_means))
