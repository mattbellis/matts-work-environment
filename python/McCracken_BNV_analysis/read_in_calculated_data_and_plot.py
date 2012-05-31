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
infile_name = sys.argv[1]
infile = np.load(infile_name)

masses = []
vtxs = []
beta0 = []
beta1 = []
beta2 = []
lambda_beta = []
meas_betas = []
gammas = []
print len(infile)
for i in xrange(6):
    masses.append(infile[i])
    # Swap out nan and inf
    masses[i][masses[i]!=masses[i]] = -999
    print "masses %d: %d" % (i,len(masses[i]))
for i in xrange(2):
    vtxs.append(infile[6+i])
    vtxs[i][vtxs[i]!=vtxs[i]] = -999
    print "vtxs %d: %d" % (i,len(vtxs[i]))
for i in xrange(2):
    beta0.append(infile[8+i])
    beta0[i][beta0[i]!=beta0[i]] = -999
    beta1.append(infile[10+i])
    beta1[i][beta1[i]!=beta1[i]] = -999
    beta2.append(infile[12+i])
    beta2[i][beta2[i]!=beta2[i]] = -999
    lambda_beta.append(infile[14+i])
    lambda_beta[i][lambda_beta[i]!=lambda_beta[i]] = -999
for i in xrange(3):
    meas_betas.append(infile[16+i])
    meas_betas[i][meas_betas[i]!=meas_betas[i]] = -999
    print "meas_betas %d: %d" % (i,len(meas_betas[i]))

################################################################################
# Pick which permutation is the ``good" permutation.
################################################################################
good_masses = [None,None,None]
bad_masses = [None,None,None]
good_vtx = None
bad_vtx = None
good_Lbeta = None
bad_Lbeta = None

for i in xrange(3):
    for permutation in xrange(2):

        i0 = 2*i
        i1 = (2*i)+1
        if permutation==1:
            i0 = (2*i)+1
            i1 = 2*i

        good_masses[i] =                          masses[i0][abs(masses[i0]-optimal_vals[i])<=abs(masses[i1]-optimal_vals[i])]
        good_masses[i] = np.append(good_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])< abs(masses[i0]-optimal_vals[i])])

        bad_masses[i] =                         masses[i0][abs(masses[i0]-optimal_vals[i])>abs(masses[i1]-optimal_vals[i])]
        bad_masses[i] = np.append(bad_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])>=abs(masses[i0]-optimal_vals[i])])

    print "masses %d" % (i)
    print len(good_masses[i])
    print len(bad_masses[i])

cut0 =     abs(masses[2]-optimal_vals[1])<=abs(masses[3]-optimal_vals[1])
cut1 =     abs(masses[3]-optimal_vals[1])< abs(masses[2]-optimal_vals[1])
anticut0 = abs(masses[2]-optimal_vals[1])> abs(masses[3]-optimal_vals[1])
anticut1 = abs(masses[3]-optimal_vals[1])>=abs(masses[2]-optimal_vals[1])

#cut0 *=     (meas_betas[0]-beta0[0]>0.04)*(meas_betas[1]-beta1[0]>0.04)*(meas_betas[2]-beta2[0]>0.04)
#cut1 *=     (meas_betas[0]-beta0[1]>0.04)*(meas_betas[1]-beta1[1]>0.04)*(meas_betas[2]-beta2[1]>0.04)

good_vtx =                    vtxs[0][cut0]
good_vtx = np.append(good_vtx,vtxs[1][cut1])

bad_vtx =                   vtxs[0][anticut0]
bad_vtx = np.append(bad_vtx,vtxs[1][anticut1])

good_beta =                     lambda_beta[0][cut0]
good_beta = np.append(good_beta,lambda_beta[1][cut1])

bad_beta =                    lambda_beta[0][anticut0]
bad_beta = np.append(bad_beta,lambda_beta[1][anticut1])

print "cuts"
print cut0
print cut1
# Try to pick out the ``correct" beta.
good_meas_betas = [[None,None,None],[None,None,None]]
good_beta0 = [None,None]
good_beta1 = [None,None]
good_beta2 = [None,None]

for i in range(0,3):
    good_meas_betas[0][i] =                                 meas_betas[i][cut0]
    good_meas_betas[0][i] = np.append(good_meas_betas[0][i],meas_betas[i][cut1])
    good_meas_betas[1][i] =                                 meas_betas[i][anticut0]
    good_meas_betas[1][i] = np.append(good_meas_betas[1][i],meas_betas[i][anticut1])

good_beta0[0] =                         beta0[0][cut0]
good_beta0[0] = np.append(good_beta0[0],beta0[1][cut1])
good_beta1[0] =                         beta1[0][cut0]
good_beta1[0] = np.append(good_beta1[0],beta1[1][cut1])
good_beta2[0] =                         beta2[0][cut0]
good_beta2[0] = np.append(good_beta2[0],beta2[1][cut1])

good_beta0[1] =                         beta0[0][anticut0]
good_beta0[1] = np.append(good_beta0[1],beta0[1][anticut1])
good_beta1[1] =                         beta1[0][anticut0]
good_beta1[1] = np.append(good_beta1[1],beta1[1][anticut1])
good_beta2[1] =                         beta2[0][anticut0]
good_beta2[1] = np.append(good_beta2[1],beta2[1][anticut1])

good_gamma = 1.0/np.sqrt(1.0-(good_beta*good_beta))
bad_gamma = 1.0/np.sqrt(1.0-(bad_beta*bad_beta))

print "beta"
print len(good_beta)
print len(bad_beta)


################################################################################
# Create the empty figures
################################################################################
print "\n"
nfigs = 5
figs = []
for i in xrange(nfigs):
    name = "fig%d" % (i)
    if i==3:
        figs.append(plt.figure(figsize=(10,9),dpi=100,facecolor='w',edgecolor='k'))
    else:
        figs.append(plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k'))

subplots = []
for i in xrange(nfigs):
    # We'll make this a nfigs x nsubplots_per_fig to store the subplots
    subplots.append([])
    if i<3:
        for j in xrange(1,4):
            subplots[i].append(figs[i].add_subplot(1,3,j))
    elif i==3:
        for j in xrange(1,7):
            subplots[i].append(figs[i].add_subplot(3,2,j))
    else:
        for j in xrange(1,3):
            subplots[i].append(figs[i].add_subplot(1,2,j))

    # Adjust the spacing between the subplots, to allow for better
    # readability.
    #figs[i].subplots_adjust(wspace=0.4,hspace=0.6,bottom=0.2)
    if i==3:
        figs[i].subplots_adjust(left=0.10, bottom=0.10, right=0.97, top=0.97, wspace=0.30, hspace=0.35)
    else:
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
#ngood = len(good_masses[0][good_vtx>4.0])
#print "ngood: %f" % (ngood/nevents)
ngood = len(good_masses[1][good_vtx>4.0])
print "ngood: %f" % (ngood/nevents)
#ngood = len(good_masses[2][good_vtx>4.0])
#print "ngood: %f" % (ngood/nevents)

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
    subplots[0][i].set_xlabel(mtitles[i])
    subplots[0][i].set_xlim(plot_ranges[i])
    subplots[0][i].set_ylim(0.0)

for i in range(0,3):
    lh.append(lch.hist_err(bad_masses[i],bins=100,range=plot_ranges[i],axes=subplots[1][i]))
    subplots[1][i].set_xlabel(mtitles[i])
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

btitles = [None,None]
charge = ['+','+','-']
for i in range(0,6):
    pindex = i/2
    index = i%2
    btitles[0] = r"particle %d (%s): $\beta_{meas} - \beta_{hyp%d}$" % (pindex,charge[pindex],index)
    btitles[1] = r"particle %d (%s): $\beta_{hyp%d}$" % (pindex,charge[pindex],index)
    if i==0 or i==1:
        lch.hist_2D(good_meas_betas[index][0]-good_beta0[index],good_beta0[index],xbins=200,ybins=200,xrange=(-1.2,1.2),yrange=(0,1.2),axes=subplots[3][i],log=True)
    elif i==2 or i==3:
        lch.hist_2D(good_meas_betas[index][1]-good_beta1[index],good_beta1[index],xbins=200,ybins=200,xrange=(-1.2,1.2),yrange=(0,1.2),axes=subplots[3][i],log=True)
    elif i==4 or i==5:
        print len(beta2)
        lch.hist_2D(good_meas_betas[index][2]-good_beta2[index],good_beta2[index],xbins=200,ybins=200,xrange=(-1.2,1.2),yrange=(0,1.2),axes=subplots[3][i],log=True)
    subplots[3][i].set_xlim(-1.2,1.2)
    subplots[3][i].set_ylim(0,1.2)
    subplots[3][i].set_xlabel(btitles[0],fontsize=15)
    subplots[3][i].set_ylabel(btitles[1],fontsize=15)
################################################################################
# Flight paths
################################################################################

delta_beta0 = good_meas_betas[0][0]-good_beta0[0]
delta_beta1 = good_meas_betas[0][1]-good_beta1[0]
delta_beta2 = good_meas_betas[0][2]-good_beta2[0]

pid_cut = (delta_beta0>-0.04)*(delta_beta1>-0.04)*(delta_beta2>-0.04)
eff = len(pid_cut[pid_cut==True])/float(len(pid_cut))
print "pid_cut:  %f"  % (eff)

print good_beta
print good_gamma
print len(good_beta)
print len(bad_beta)
c = 1.0
#c = 29.97
#lh.append(lch.hist_err(good_vtx/(good_beta*good_gamma*c),bins=100,range=(0.0,30.00),axes=subplots[4][0]))
#lh.append(lch.hist_err(bad_vtx/(bad_beta*bad_gamma*c),bins=100,range=(0.0,30.00),axes=subplots[4][1]))
lh.append(lch.hist_err(good_vtx*(good_gamma),bins=100,range=(0.0,100.00),axes=subplots[4][0]))
lh.append(lch.hist_err(bad_vtx*(bad_gamma),bins=100,range=(0.0,100.00),axes=subplots[4][1]))
subplots[4][0].set_xlabel(r"$\Lambda$ flight length (best)",fontsize=20)
subplots[4][1].set_xlabel(r"$\Lambda$ flight length (worst)",fontsize=20)
################################################################################
subplots[4][0].set_xlim(0.0)
subplots[4][1].set_xlim(0.0)
subplots[4][0].set_ylim(0.0)
subplots[4][1].set_ylim(0.0)

#subplots[4][0].set_yscale('log')

basename = infile_name.split('/')[-1].split('.')[0]
for i in xrange(nfigs):
    figure_filename = "Plots/%s_%s.png" % (basename,i)
    print "Saving %s..." % (figure_filename)
    figs[i].savefig(figure_filename)

plt.show()

