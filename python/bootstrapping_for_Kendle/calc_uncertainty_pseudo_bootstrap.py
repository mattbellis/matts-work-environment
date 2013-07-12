#!/usr/bin/env python

from math import *
import numpy as np
import sys
import random as rnd 

################################################################################
# Read in the values from the file.
################################################################################

outfile = open("test_output.txt","w+")

#file = open(sys.argv[1])
#n_subsamples = 5

################################################################################
# Kendle, this is where you need to change things.
################################################################################
#file = open("FILENAME") # Input filename
file = open("kendles_thumb_drive/Exp3/exp3_grain3_KF.txt") # Input filename
column_number = 2
n_subsamples = 9 # How big are the subsamples do you want to do?
do_replacement = False # Set this to be True if you want to do the full bootstrap technique
max = 10000 # The number of subsamples you want to generate.

pts = []
count = 0
for line in file:
    vals = line.split()
    if len(vals)>5 and line.find('ELEM')<0:
        pts.append(float(vals[column_number]))
    count += 1

npts = len(pts)

x = []
x = pts

#ncounts = 1000

#npts = len(x)

print "npts: %d\t\t%f +/- %f" % (npts, np.mean(pts), np.std(pts))

#exit()


################################################################################
# Try the jackknife technique
################################################################################

ncounts = 1

print "Starting bootstrap calculations....."

sub_means = []

#n_subsamples = 5
'''
if len(sys.argv)>=3:
    n_subsamples = int(sys.argv[2])
'''

'''
do_replacement = False
if len(sys.argv)==4:
    if sys.argv[3] == 'replacement':
        do_replacement = True
'''

for k in xrange(max):

    x = []
    if do_replacement is False:
       # Without replacement
       x = rnd.sample(pts,n_subsamples)
    else:
       # With replacement
       for i in xrange(n_subsamples):
           index = rnd.randint(0,npts-1)
           x.append(pts[index])

    sub_mean = np.mean(x)
    sub_err = np.std(x)

    #print "---------"
    #print x
    sub_means.append(sub_mean)

    output = "%02d %f +/- %f\n" % (k,sub_mean,sub_err)
    outfile.write(output)
    print output

outfile.close()

#print sub_means
print "From entire sample: %f +/- %f" % (np.mean(sub_means), np.std(sub_means))
