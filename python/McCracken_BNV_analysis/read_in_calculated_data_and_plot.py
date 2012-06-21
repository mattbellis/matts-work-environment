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
bad_perm_masses = [np.array([]),np.array([]),np.array([])]
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
good_perm_masses = [None,None,None]
bad_perm_masses = [None,None,None]
good_masses = [None,None,None]
bad_masses = [None,None,None]
good_vtx = None
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

        good_perm_masses[i] =                          masses[i0][abs(masses[i0]-optimal_vals[i])<=abs(masses[i1]-optimal_vals[i])]
        good_perm_masses[i] = np.append(good_perm_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])< abs(masses[i0]-optimal_vals[i])])

        bad_perm_masses[i] =                         masses[i0][abs(masses[i0]-optimal_vals[i])>abs(masses[i1]-optimal_vals[i])]
        bad_perm_masses[i] = np.append(bad_perm_masses[i],masses[i1][abs(masses[i1]-optimal_vals[i])>=abs(masses[i0]-optimal_vals[i])])

    print "masses %d" % (i)
    print len(good_perm_masses[i])
    print len(bad_perm_masses[i])

# Cut on invariant mass of Lambda candidate to select correct permutation
cut0 =     abs(masses[2]-optimal_vals[1])<=abs(masses[3]-optimal_vals[1])
cut1 =     abs(masses[3]-optimal_vals[1])< abs(masses[2]-optimal_vals[1])
anticut0 = abs(masses[2]-optimal_vals[1])> abs(masses[3]-optimal_vals[1])
anticut1 = abs(masses[3]-optimal_vals[1])>=abs(masses[2]-optimal_vals[1])

#cut0 *=     (meas_betas[0]-beta0[0]>0.04)*(meas_betas[1]-beta1[0]>0.04)*(meas_betas[2]-beta2[0]>0.04)
#cut1 *=     (meas_betas[0]-beta0[1]>0.04)*(meas_betas[1]-beta1[1]>0.04)*(meas_betas[2]-beta2[1]>0.04)

print cut0

good_masses[0] =                       masses[0][cut0==True]
good_masses[0] = np.append(good_masses[0],masses[1][cut1==True])
good_masses[1] =                       masses[2][cut0==True]
good_masses[1] = np.append(good_masses[1],masses[3][cut1==True])
good_masses[2] =                       masses[4][cut0==True]
good_masses[2] = np.append(good_masses[2],masses[5][cut1==True])

bad_masses[0] =                      masses[0][anticut0==True]
bad_masses[0] = np.append(bad_masses[0],masses[1][anticut1==True])
bad_masses[1] =                      masses[2][anticut0==True]
bad_masses[1] = np.append(bad_masses[1],masses[3][anticut1==True])
bad_masses[2] =                      masses[4][anticut0==True]
bad_masses[2] = np.append(bad_masses[2],masses[5][anticut1==True])


good_vtx =                    vtxs[0][cut0==True]
good_vtx = np.append(good_vtx,vtxs[1][cut1==True])

bad_vtx =                   vtxs[0][anticut0==True]
bad_vtx = np.append(bad_vtx,vtxs[1][anticut1==True])

good_lambda_beta =                     lambda_beta[0][cut0==True]
good_lambda_beta = np.append(good_lambda_beta,lambda_beta[1][cut1==True])

bad_beta =                    lambda_beta[0][anticut0==True]
bad_beta = np.append(bad_beta,lambda_beta[1][anticut1==True])

print "cuts"
print cut0
print cut1
# Try to pick out the ``correct" beta.
good_meas_betas = [[None,None,None],[None,None,None]]
good_beta0 = [None,None]
good_beta1 = [None,None]
good_beta2 = [None,None]

for i in range(0,3):
    good_meas_betas[0][i] =                                 meas_betas[i][cut0==True]
    good_meas_betas[0][i] = np.append(good_meas_betas[0][i],meas_betas[i][cut1==True])
    good_meas_betas[1][i] =                                 meas_betas[i][anticut0==True]
    good_meas_betas[1][i] = np.append(good_meas_betas[1][i],meas_betas[i][anticut1==True])

good_beta0[0] =                         beta0[0][cut0==True]
good_beta0[0] = np.append(good_beta0[0],beta0[1][cut1==True])
good_beta1[0] =                         beta1[0][cut0==True]
good_beta1[0] = np.append(good_beta1[0],beta1[1][cut1==True])
good_beta2[0] =                         beta2[0][cut0==True]
good_beta2[0] = np.append(good_beta2[0],beta2[1][cut1==True])

good_beta0[1] =                         beta0[0][anticut0==True]
good_beta0[1] = np.append(good_beta0[1],beta0[1][anticut1==True])
good_beta1[1] =                         beta1[0][anticut0==True]
good_beta1[1] = np.append(good_beta1[1],beta1[1][anticut1==True])
good_beta2[1] =                         beta2[0][anticut0==True]
good_beta2[1] = np.append(good_beta2[1],beta2[1][anticut1==True])

good_lambda_gamma = 1.0/np.sqrt(1.0-(good_lambda_beta*good_lambda_beta))
bad_gamma = 1.0/np.sqrt(1.0-(bad_beta*bad_beta))

print "beta"
print len(good_lambda_beta)
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

nevents = float(len(good_perm_masses[0]))
print "nevents: %f" % (nevents)

print 'Good masses in limits'
print len(good_perm_masses[0][(good_perm_masses[0]>limits[0][0])*(good_perm_masses[0]<limits[0][1])])
print len(good_perm_masses[1][(good_perm_masses[1]>limits[1][0])*(good_perm_masses[1]<limits[1][1])])
print len(good_perm_masses[2][(good_perm_masses[2]>limits[2][0])*(good_perm_masses[2]<limits[2][1])])

print "\n"
#ngood = len(good_perm_masses[0][good_vtx>4.0])
#print "ngood: %f" % (ngood/nevents)
ngood = len(good_perm_masses[1][good_vtx>8.0])
print "ngood: %f" % (ngood/nevents)
#ngood = len(good_perm_masses[2][good_vtx>4.0])
#print "ngood: %f" % (ngood/nevents)

print "\n"
print 'Bad masses in limits'
print len(bad_perm_masses[0][(bad_perm_masses[0]>limits[0][0])*(bad_perm_masses[0]<limits[0][1])])
print len(bad_perm_masses[1][(bad_perm_masses[1]>limits[1][0])*(bad_perm_masses[1]<limits[1][1])])
print len(bad_perm_masses[2][(bad_perm_masses[2]>limits[2][0])*(bad_perm_masses[2]<limits[2][1])])

print "\n"
print "vtx"
print len(vtxs[0])
print len(good_vtx[good_vtx>8.0])/float(len(vtxs[0]))
print len(bad_vtx[bad_vtx>8.0])/float(len(vtxs[0]))

lh = []

################################################################################
# Plot them
################################################################################
plot_ranges = [(-0.1,0.10),(1.0,1.50),(1.0,1.50)]

for i in range(0,3):
    lh.append(lch.hist_err(good_perm_masses[i],bins=100,range=plot_ranges[i],axes=subplots[0][i]))
    subplots[0][i].set_xlabel(mtitles[i])
    subplots[0][i].set_xlim(plot_ranges[i])
    subplots[0][i].set_ylim(0.0)

for i in range(0,3):
    lh.append(lch.hist_err(bad_perm_masses[i],bins=100,range=plot_ranges[i],axes=subplots[1][i]))
    subplots[1][i].set_xlabel(mtitles[i])
    subplots[1][i].set_xlim(plot_ranges[i])
    subplots[1][i].set_ylim(0.0)


lh2d = []
xindex = [0,0,1]
yindex = [1,2,2]

print len(good_perm_masses[0])
print len(good_perm_masses[1])
print len(good_perm_masses[2])

print len(bad_perm_masses[0])
print len(bad_perm_masses[1])
print len(bad_perm_masses[2])

for i in range(0,3):
    i0 = xindex[i]
    i1 = yindex[i]
    lh2d.append(lch.hist_2D(good_perm_masses[i0],good_perm_masses[i1],xbins=100,ybins=100,xrange=plot_ranges[i0],yrange=plot_ranges[i1],axes=subplots[2][i],log=True))
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

print good_lambda_beta
print good_lambda_gamma
print len(good_lambda_beta)
print len(bad_beta)
c = 1.0
#c = 29.97
#lh.append(lch.hist_err(good_vtx/(good_lambda_beta*good_lambda_gamma*c),bins=100,range=(0.0,30.00),axes=subplots[4][0]))
#lh.append(lch.hist_err(bad_vtx/(bad_beta*bad_gamma*c),bins=100,range=(0.0,30.00),axes=subplots[4][1]))
#lh.append(lch.hist_err(good_vtx*(good_lambda_gamma),bins=100,range=(0.0,100.00),axes=subplots[4][0]))
#lh.append(lch.hist_err(bad_vtx*(bad_gamma),bins=100,range=(0.0,100.00),axes=subplots[4][1]))

h0,xpts0,ypts0,xpts_err0,ypts_err0 = lch.hist_err(good_vtx,bins=100,range=(0.0,100.00),axes=subplots[4][0])
h1,xpts1,ypts1,xpts_err1,ypts_err1 = lch.hist_err(bad_vtx,bins=100,range=(0.0,100.00),axes=subplots[4][1])

subplots[4][0].set_xlabel(r"$\Lambda$ flight length (best)",fontsize=20)
subplots[4][1].set_xlabel(r"$\Lambda$ flight length (worst)",fontsize=20)
################################################################################
subplots[4][0].set_xlim(0.0)
subplots[4][1].set_xlim(0.0)
subplots[4][0].set_ylim(0.0,1.1*max(ypts0))
subplots[4][1].set_ylim(0.0,1.1*max(ypts1))

#subplots[4][0].set_yscale('log')

basename = infile_name.split('/')[-1].split('.')[0]
for i in xrange(nfigs):
    figure_filename = "Plots/%s_%s.png" % (basename,i)
    print "Saving %s..." % (figure_filename)
    figs[i].savefig(figure_filename)

################################################################################
# Count some values.
################################################################################

outfile_name = "effects_of_cuts/%s" % (infile_name.split('/')[-1].split('.')[0])

# Test cuts 
#nslices = 2
#cut_tot_missing_mass = np.linspace(1.010,0.001,nslices)
#cut_inv_lambda_mass = np.linspace(1.050,0.005,nslices)
#cut_flight_len = np.linspace(-1.00,10.00,nslices)
#cut_beta_pid = np.linspace(1.05,0.005,nslices)
#cut_missing_mass_off_k = np.linspace(1.000,0.040,2)

# Real cuts
nslices = 3
#cut_tot_missing_mass = np.linspace(0.010,0.001,nslices)
cut_tot_missing_mass = np.array([0.5,0.01,0.05,0.02,0.001])
#cut_inv_lambda_mass = np.linspace(0.050,0.005,nslices)
cut_inv_lambda_mass = np.array([0.5,0.05,0.03,0.02,0.01,0.05])
cut_flight_len = np.linspace(0.00,4.00,nslices)
#cut_beta_pid = np.linspace(0.05,0.005,nslices)
#cut_beta_pid = np.array([1.0,0.5,0.4,0.3,0.2,0.1,0.05,0.03,0.01,0.005])
cut_beta_pid = np.array([1.0,0.5,0.2,0.1,0.05,0.02])
cut_left_beta = -0.05
cut_missing_mass_off_k = np.linspace(1.000,0.040,2)

index = [None,None,None,None,None,None,None] # 7?
ncuts = len(index)

nmatrix = len(cut_tot_missing_mass)*len(cut_inv_lambda_mass)*len(cut_flight_len)*(len(cut_beta_pid)**3)*len(cut_missing_mass_off_k)

print "nmatrix: ",nmatrix

cuts = np.zeros((ncuts,int(nmatrix)))
tot = np.zeros(int(nmatrix))
remain = np.zeros(int(nmatrix))

#exit()
        

tot_events = len(good_masses[0])
print "tot: ",tot_events
i = 0
for c0 in cut_tot_missing_mass:
    index[0] = abs(good_masses[0]-0.000)<c0
    print c0
    for c1 in cut_inv_lambda_mass:
        index[1] = abs(good_masses[1]-mass_L)<c1
        for c2 in cut_beta_pid:
            #index[2] = abs(delta_beta0)<c2
            index[2]  = delta_beta0<c2
            index[2] *= delta_beta0>cut_left_beta
            for c3 in cut_beta_pid:
                #index[3] = abs(delta_beta1)<c3
                index[3]  = delta_beta1<c3
                index[3] *= delta_beta1>cut_left_beta
                for c4 in cut_beta_pid:
                    #index[4] = abs(delta_beta2)<c4
                    index[4]  = delta_beta2<c4
                    index[4] *= delta_beta2>cut_left_beta
                    for c5 in cut_flight_len:
                        index[5] = good_vtx>c5
                        for c6 in cut_missing_mass_off_k:
                            index[6] = abs(good_masses[2]-mass_L)<c6

                            master_index = index[0]*index[1]*index[2]*index[3]*index[4]*index[5]*index[6]

                            remaining_events = len(master_index[master_index==True])

                            tot[i] = tot_events
                            remain[i] = remaining_events

                            cuts[0][i] = c0
                            cuts[1][i] = c1
                            cuts[2][i] = c2
                            cuts[3][i] = c3
                            cuts[4][i] = c4
                            cuts[5][i] = c5
                            cuts[6][i] = c6

                            #output = "%4.2f %4.2f %4.2f %4.2f %4.2f %6.2f %4.2f\t" % (c0,c1,c2,c3,c4,c5,c6) 
                            #output += "%8.1f %8.1f"  % (tot_events,remaining_events)
                            #print output

                            i += 1

outarrays = [cuts,tot,remain]
np.save(outfile_name,outarrays)

#plt.show()

