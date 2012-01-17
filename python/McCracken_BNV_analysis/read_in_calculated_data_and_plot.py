#!/usr/bin/env python

import sys
from math import *

import numpy as np
from StringIO import StringIO

import matplotlib.pyplot as plt
import csv

################################################################################

#infile = csv.reader(sys.argv[1],'rb')
infile = open(sys.argv[1],'r')

n = 0
masses = [[],[],[]]
flight_length = []
count = 0
beam = None
nvtx = 0

for line in infile:
    #vals = line.split()
    vals = np.genfromtxt(StringIO(line),dtype=(float),delimiter=",")

    if abs(vals[0])<abs(vals[3]):
        masses[0].append(-vals[0]*vals[0])
    else:
        masses[0].append(vals[3]*vals[3])

    if abs(vals[1]-1.115)<abs(vals[4]-1.115):
        masses[1].append(vals[1])
    else:
        masses[1].append(vals[4])

    if abs(vals[2]-1.115)<abs(vals[5]-1.115):
        masses[2].append(vals[2])
    else:
        masses[2].append(vals[5])

    count +=1 
    if count > 1000000:
        break

nfigs = 1
figs = []
for i in xrange(nfigs):
    figs.append(plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k'))

subplots = []
for i in xrange(nfigs):
    # We'll make this a nfigs x nsubplots_per_fig to store the subplots
    subplots.append([])
    for j in xrange(1,4):
        # Divide the figure into a 2x2 grid.
        subplots[i].append(figs[i].add_subplot(1,3,j))

    # Adjust the spacing between the subplots, to allow for better
    # readability.
    figs[i].subplots_adjust(wspace=0.4,hspace=0.6,bottom=0.2)



#print masses
h0 = subplots[0][0].hist(masses[0],100,range=(-0.1,0.10),histtype='stepfilled',color='blue')
subplots[0][0].set_xlabel('Total missing mass')

h1 = subplots[0][1].hist(masses[1],100,range=(1.0,1.50),histtype='stepfilled',color='grey')
subplots[0][1].set_xlabel(r"Missing mass off the $K^{+}$")

h2 = subplots[0][2].hist(masses[2],100,range=(1.0,1.50),histtype='stepfilled',color='red',alpha=0.5)
subplots[0][2].set_xlabel(r"Invariant mass of the $p \pi^{-}$")

figs[0].savefig("k_lambda.png")

plt.show()

