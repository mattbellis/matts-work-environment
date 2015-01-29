#!/usr/bin/env python

from math import *
import sys
import numpy
import random

# Open the file
infilename = sys.argv[1]
infile = open(infilename)

xlo = [5.2,-0.2,0.7]
xhi = [5.3, 0.2,1.0]

# Get all the values
x = [[], [], []]
n = 0
for line in infile:
    vals = line.split()

    # Check that the event is within our ranges
    good_event = True
    for i,v in enumerate(vals):
        z = float(v)
        if z<xlo[i] or z>xhi[i]:
            good_event = False

    if good_event:
        x[0].append(float(vals[0]))
        x[1].append(float(vals[1]))
        x[2].append(float(vals[2]))
        n+=1

    if n==5000:
        break


nentries = len(x[0])

# Get mean and sigma
mean = []
sigma = []
for i in range(0,3):
    mean.append(numpy.mean(x[i]))
    sigma.append(numpy.std(x[i]))

print mean
print sigma

# Calculate correlation coefficients
cc = []
index = []
for i in range(0,3):
    for j in range(i+1,3):
        cc.append(numpy.corrcoef(x[i], x[j])[0][1])
        index.append([i,j])

for i,c in enumerate(cc):
    print index[i]
    print c

'''
for i in range(0,3):
    for j in range(0,3):
        mycc = 0.0
        for k in range(0,nentries):
            mycc += (x[i][k]-mean[i])*(x[j][k]-mean[j])/((nentries-1)*sigma[i]*sigma[j])
        print mycc
'''

# Let's try to calculate the errors by bootstrapping
many_cc = [[], [], []]
sample_pct = 0.9
sample_size = int(sample_pct*nentries)

ntrials = nentries-sample_size

print sample_size
print ntrials
for n in range(0,ntrials):

    # Create the subsamples.
    # There will be sample_size entries in them.
    xtest = [[],[],[]]
    indices = random.sample(xrange(nentries),sample_size)
    for j in range(0,sample_size):
        for i in range(0,3):
            xtest[i].append(x[i][indices[j]])
    
    #print " ----------------------------------- "
    #print xtest[0][0]
    #print "xtest0: %6.4f %6.4f" % (numpy.mean(xtest[0]), numpy.std(xtest[0]))
    #print "xtest1: %6.4f %6.4f" % (numpy.mean(xtest[1]), numpy.std(xtest[1]))
    #print "xtest2: %6.4f %6.4f" % (numpy.mean(xtest[2]), numpy.std(xtest[2]))
    count = 0
    for i in range(0,3):
        for j in range(i+1,3):
            #print "test vals: %f %f " % (xtest[i][0],xtest[j][0])
            many_cc[count].append(numpy.corrcoef(xtest[i], xtest[j])[0][1])
            '''
            if i==0 and j==1:
                print "---------"
                print many_cc[count][n]
            '''
            count += 1



print mean
print sigma

#print many_cc[2]

for i,c in enumerate(cc):
    print index[i]
    print c

print "----------------------"
for i in range(0,3):
    print "%6.4f %6.4f %6.4f" % (cc[i], numpy.mean(many_cc[i]), numpy.std(many_cc[i]))
        




