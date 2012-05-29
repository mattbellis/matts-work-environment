#!/usr/bin/env python

import sys
from math import *

import numpy as np
from StringIO import StringIO

import matplotlib.pyplot as plt
import csv

import lichen.lichen as lch

from analysis_utilities import *

################################################################################

#infile = csv.reader(sys.argv[1],'rb')
#infile = open(sys.argv[1],'r')

n = 0
masses = [np.array([]),np.array([]),np.array([])]
bad_masses = [np.array([]),np.array([]),np.array([])]
vtxs = [np.array([]),np.array([])]
flight_length = []
count = 0
beam = None
nvtx = 0

limits = [[-0.005,0.005],[1.1060,1.160],[1.1060,1.160]]
optimal_vals = [0.0,mass_L,mass_L]

max_events = 10000

mtitles = []
mtitles.append(r'Total missing mass squared')
mtitles.append(r"Invariant mass of the $\Lambda$ candidate")
mtitles.append(r"Missing mass off the $K^{+}$ candidate")

################################################################################
# Read in the data
################################################################################
infile = np.load(sys.argv[1])

masses = []
vtxs = []
for i in xrange(6):
    masses.append(infile[i])
    # Swap out nan and inf
    masses[i][masses[i]!=masses[i]] = -999
    print "masses %d: %d" % (i,len(masses[i]))
for i in xrange(2):
    vtxs.append(infile[6+i])
    # Swap out nan and inf
    vtxs[i][vtxs[i]!=vtxs[i]] = -999
    print "vtxs %d: %d" % (i,len(vtxs[i]))

################################################################################
# Pick which permutation is the ``good" permutation.
################################################################################
good_masses = [None,None,None]
bad_masses = [None,None,None]
good_vtx = None
bad_vtx = None

for i in xrange(3):
    for permutation in xrange(2):

        i0 = 2*i
        i1 = (2*i)+1
        if permutation==1:
            i0 = (2*i)+1
            i1 = 2*i

        good_masses[i] =           masses[i0][abs(masses[i0]-optimal_vals[i])<abs(masses[i1]-optimal_vals[i])]
        good_masses[i] = np.append(good_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])<=abs(masses[i0]-optimal_vals[i])])

        bad_masses[i] =           masses[i0][abs(masses[i0]-optimal_vals[i])>=abs(masses[i1]-optimal_vals[i])]
        bad_masses[i] = np.append(bad_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])>=abs(masses[i0]-optimal_vals[i])])

good_vtx =                    vtxs[0][abs(masses[0]-optimal_vals[0])<abs(masses[1]-optimal_vals[0])]
good_vtx = np.append(good_vtx,vtxs[1][abs(masses[1]-optimal_vals[0])<abs(masses[0]-optimal_vals[0])])

bad_vtx =                   vtxs[0][abs(masses[0]-optimal_vals[0])>=abs(masses[1]-optimal_vals[0])]
bad_vtx = np.append(bad_vtx,vtxs[1][abs(masses[1]-optimal_vals[0])>=abs(masses[0]-optimal_vals[0])])


################################################################################
# Create the empty figures
################################################################################
print "\n"
nfigs = 4
figs = []
for i in xrange(nfigs):
    name = "fig%d" % (i)
    figs.append(plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k'))

subplots = []
for i in xrange(nfigs):
    # We'll make this a nfigs x nsubplots_per_fig to store the subplots
    subplots.append([])
    for j in xrange(1,4):
        if (i<3):
            subplots[i].append(figs[i].add_subplot(1,3,j))
        else:
            if j<3:
                subplots[i].append(figs[i].add_subplot(1,2,j))

    # Adjust the spacing between the subplots, to allow for better
    # readability.
    #figs[i].subplots_adjust(wspace=0.4,hspace=0.6,bottom=0.2)
    figs[i].subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.30, hspace=None)


################################################################################

################################################################################
# Count some numbers
################################################################################

nevents = float(len(good_masses[0]))
print "nevents: %f" % (nevents)

print 'Good masses in limits'
print len(good_masses[0][(good_masses[0]>limits[0][0])*(good_masses[0]<limits[0][1])])
print len(good_masses[1][(good_masses[1]>limits[1][0])*(good_masses[1]<limits[1][1])])
print len(good_masses[2][(good_masses[2]>limits[2][0])*(good_masses[2]<limits[2][1])])

print "\n"
ngood = len(good_masses[0][good_vtx>4.0])
print "ngood: %f" % (ngood/nevents)
ngood = len(good_masses[1][good_vtx>4.0])
print "ngood: %f" % (ngood/nevents)
ngood = len(good_masses[2][good_vtx>4.0])
print "ngood: %f" % (ngood/nevents)

print "\n"
print 'Bad masses in limits'
print len(bad_masses[0][(bad_masses[0]>limits[0][0])*(bad_masses[0]<limits[0][1])])
print len(bad_masses[1][(bad_masses[1]>limits[1][0])*(bad_masses[1]<limits[1][1])])
print len(bad_masses[2][(bad_masses[2]>limits[2][0])*(bad_masses[2]<limits[2][1])])

print "\n"
print "vtx"
print len(vtxs[0])
print len(good_vtx[good_vtx>4.0])/float(len(vtxs[0]))
print len(bad_vtx[bad_vtx>4.0])/float(len(vtxs[0]))

lh = []

################################################################################
# Plot them
################################################################################
plot_ranges = [(-0.1,0.10),(1.0,1.50),(1.0,1.50)]

for i in range(0,3):
    lh.append(lch.hist_err(good_masses[i],bins=100,range=plot_ranges[i],axes=subplots[0][i]))
    subplots[0][i].set_xlabel(mtitles[0])
    subplots[0][i].set_xlim(plot_ranges[i])
    subplots[0][i].set_ylim(0.0)

for i in range(0,3):
    lh.append(lch.hist_err(bad_masses[i],bins=100,range=plot_ranges[i],axes=subplots[1][i]))
    subplots[1][i].set_xlabel(mtitles[0])
    subplots[1][i].set_xlim(plot_ranges[i])
    subplots[1][i].set_ylim(0.0)


lh2d = []
xindex = [0,0,1]
yindex = [1,2,2]

print len(good_masses[0])
print len(good_masses[1])
print len(good_masses[2])

print len(bad_masses[0])
print len(bad_masses[1])
print len(bad_masses[2])

for i in range(0,3):
    i0 = xindex[i]
    i1 = yindex[i]
    lh2d.append(lch.hist_2D(good_masses[i0],good_masses[i1],xbins=100,ybins=100,xrange=plot_ranges[i0],yrange=plot_ranges[i1],axes=subplots[2][i]))
    subplots[2][i].set_xlabel(mtitles[i0])
    subplots[2][i].set_ylabel(mtitles[i1])
    #subplots[2][0].set_aspect('auto')
    #figs[2].add_subplot(1,3,1)
    #plt.colorbar(cax=subplots[2][0])

################################################################################
# Flight paths
################################################################################

lh.append(lch.hist_err(vtxs[0],bins=100,range=(0.0,30.00),axes=subplots[3][0]))
lh.append(lch.hist_err(vtxs[1],bins=100,range=(0.0,30.00),axes=subplots[3][1]))
subplots[3][0].set_ylim(0.0)
subplots[3][1].set_ylim(0.0)

figs[0].savefig("Plots/fig0.png")
figs[1].savefig("Plots/fig1.png")
figs[2].savefig("Plots/fig2.png")
figs[3].savefig("Plots/fig3.png")

plt.show()

