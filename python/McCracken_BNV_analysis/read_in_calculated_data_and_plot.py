#!/usr/bin/env python

import sys
from math import *

import numpy as np
from StringIO import StringIO

import matplotlib.pyplot as plt
import csv

import lichen.lichen as lch

################################################################################

#infile = csv.reader(sys.argv[1],'rb')
infile = open(sys.argv[1],'r')

n = 0
masses = [np.array([]),np.array([]),np.array([])]
bad_masses = [np.array([]),np.array([]),np.array([])]
vtxs = [np.array([]),np.array([])]
flight_length = []
count = 0
beam = None
nvtx = 0

max_events = 1000

mtitles = []
mtitles.append(r"Invariant mass of the $\Lambda$ candidate")
mtitles.append(r'Total missing mass squared')
mtitles.append(r"Missing mass off the $K^{+}$ candidate")

for line in infile:
    #vals = line.split()
    vals = np.genfromtxt(StringIO(line),dtype=(float),delimiter=" ")

    no_nans = True
    for v in vals:
        if v!=v:
            no_nans = False
            break;

    if no_nans:

        if abs(vals[0]-1.115)<abs(vals[3]-1.115):
            masses[0] = np.append(masses[0],vals[0])
            bad_masses[0] = np.append(bad_masses[0],vals[3])
        else:
            masses[0] = np.append(masses[0],vals[3])
            bad_masses[0] = np.append(bad_masses[0],vals[0])

        if abs(vals[1])<abs(vals[4]):
            masses[1] = np.append(masses[1],vals[1])
            bad_masses[1] = np.append(bad_masses[1],vals[4])
        else:
            masses[1] = np.append(masses[1],vals[4])
            bad_masses[1] = np.append(bad_masses[1],vals[1])

        if abs(vals[2]-1.115)<abs(vals[5]-1.115):
            masses[2] = np.append(masses[2],vals[2])
            bad_masses[2] = np.append(bad_masses[2],vals[5])
        else:
            masses[2] = np.append(masses[2],vals[5])
            bad_masses[2] = np.append(bad_masses[2],vals[2])

    count +=1 
    if count>max_events:
        break
    if count%10000==0:
        print count


print "\n"
nfigs = 3
figs = []
for i in xrange(nfigs):
    name = "fig%d" % (i)
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
    #figs[i].subplots_adjust(wspace=0.4,hspace=0.6,bottom=0.2)
    figs[i].subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.30, hspace=None)



#print masses
print 'Masses'
print len(masses[0])
print len(masses[1])
print len(masses[2])

print "Subcategories"
#m=masses[0]
print len(masses[0][masses[0]>1.100 and masses[0]<1.130])

print 'Bad masses'
print len(bad_masses[0])
print len(bad_masses[1])
print len(bad_masses[2])

lh = []

# Correct assignments
lh.append(lch.hist_err(masses[0],bins=100,range=(1.0,1.50),axes=subplots[0][0]))
subplots[0][0].set_xlabel(mtitles[0])
subplots[0][0].set_xlim(1.0,1.50)

lh.append(lch.hist_err(masses[1],bins=100,range=(-0.1,0.10),axes=subplots[0][1]))
subplots[0][1].set_xlabel(mtitles[1])
subplots[0][1].set_xlim(-0.1,0.1)

lh.append(lch.hist_err(masses[2],bins=100,range=(1.0,1.50),axes=subplots[0][2]))
subplots[0][2].set_xlabel(mtitles[2])
subplots[0][2].set_xlim(1.0,1.50)

# Incorrect assignments
lh.append(lch.hist_err(bad_masses[0],bins=100,range=(1.0,1.50),axes=subplots[1][0]))
subplots[1][0].set_xlabel(mtitles[0])
subplots[1][0].set_xlim(1.0,1.50)

lh.append(lch.hist_err(bad_masses[1],bins=100,range=(-0.1,0.10),axes=subplots[1][1]))
subplots[1][1].set_xlabel(mtitles[1])
subplots[1][1].set_xlim(-0.1,0.1)

lh.append(lch.hist_err(bad_masses[2],bins=100,range=(1.0,1.50),axes=subplots[1][2]))
subplots[1][2].set_xlabel(mtitles[2])
subplots[1][2].set_xlim(1.0,1.50)

#h3 = subplots[1][0].hist(vtxs[0],100,range=(0.0,20.00),histtype='stepfilled',color='red',alpha=0.5)
#subplots[1][0].set_xlabel(r"Secondary vertex displacement $p \pi^{-}$")

#h4 = subplots[1][1].hist(vtxs[1],100,range=(0.0,20.00),histtype='stepfilled',color='red',alpha=0.5)
#subplots[1][1].set_xlabel(r"Secondary vertex displacement $p \pi^{-}$")

lh2d0 = lch.hist_2D(masses[0],masses[2],xbins=100,ybins=100,xrange=[0.90,1.40],yrange=[0.90,1.40],axes=subplots[2][0])
subplots[2][0].set_xlabel(mtitles[0])
subplots[2][0].set_ylabel(mtitles[2])
#subplots[2][0].set_aspect('auto')
#figs[2].add_subplot(1,3,1)
#plt.colorbar(cax=subplots[2][0])

lh2d1 = lch.hist_2D(masses[0],masses[1],xbins=100,ybins=100,xrange=[0.90,1.40],yrange=[-0.10,0.1],axes=subplots[2][1])
subplots[2][1].set_xlabel(mtitles[0])
subplots[2][1].set_ylabel(mtitles[1])
#subplots[2][1].set_aspect('auto')
#figs[2].add_subplot(1,3,2)
#plt.colorbar()

lh2d2 = lch.hist_2D(masses[1],masses[2],xbins=100,ybins=100,xrange=[-0.10,0.1],yrange=[0.90,1.40],axes=subplots[2][2])
subplots[2][2].set_xlabel(mtitles[1])
subplots[2][2].set_ylabel(mtitles[2])
#subplots[2][2].set_aspect('auto')
#figs[2].add_subplot(1,3,3)
#plt.colorbar()

figs[0].savefig("Plots/fig0.png")
figs[1].savefig("Plots/fig1.png")
figs[2].savefig("Plots/fig2.png")

plt.show()

