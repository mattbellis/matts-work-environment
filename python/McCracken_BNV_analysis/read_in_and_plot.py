#!/usr/bin/env python

import sys
from math import *

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Calculate the magnitude of a three-vector
################################################################################
def magnitude_of_3vec(v3):

    magnitude = sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

    return magnitude

################################################################################
# Calculate the mass of a particle using Classical Physics
################################################################################
def mass_from_classical_physics(v4):

    pmag = magnitude_of_3vec(v4[1:4])

    mass = pmag*pmag/(2.0*v4[0])

    return mass

################################################################################
# Calculate the mass of a particle using Special Relativity
################################################################################
def mass_from_special_relativity(v4):

    pmag = magnitude_of_3vec(v4[1:4])

    mass_squared = v4[0]*v4[0] - pmag*pmag

    if mass_squared>0:
        return sqrt(mass_squared)
    else:
        return -sqrt(abs(mass_squared))

################################################################################

infile = open(sys.argv[1])

mass_e = 0.000511
mass_pi = 0.139570
mass_k = 0.493677
mass_p = 0.938272

target = np.array([mass_p,0.0,0.0,0.0])

p = [0.0, 0.0, 0.0, 0.0, 0.0]
n = 0
masses = []
masses_inv = []
masses_mm = []
flight_length = []
vtx = [0.0,0.0,0.0]
count = 0
beam = None
nvtx = 0
for line in infile:
    vals = line.split()
    if len(vals)==1:

        beam_e = float(vals[0])
        beam = np.array([beam_e, 0.0, 0.0, beam_e])

    elif len(vals)==5:
        #print vals

        #v3 = [float(vals[0]),float(vals[1]),float(vals[2])]
        x = [float(vals[0]),float(vals[1]),float(vals[2])]
        v3 = np.array(x)

        energy = 0
        pmag = magnitude_of_3vec(v3)
        if n==0:
            mass = mass_k
        elif n==1:
            #mass = mass_e
            #mass = mass_k
            mass = mass_p
        elif n==2:
            #mass = mass_k
            #mass = mass_e
            mass = mass_pi
        energy = sqrt(mass*mass + pmag*pmag)

        p[n] = np.insert(v3,[0],energy)
        #print p[n]
        #print p[n]

        n += 1

        if n>=3:
            ############ Do some calculations #######################
            print beam
            p4 = beam + target - p[0]
            masses_mm.append(mass_from_special_relativity(p4))

            p4 = p[1]+p[2]
            masses_inv.append(mass_from_special_relativity(p4))

            n=0
            count +=1 
            if count > 1000000:
                break

    elif len(vals)==3:

        x = [float(vals[0]),float(vals[1]),float(vals[2])]
        v3 = np.array(x)

        vtx[nvtx] = v3

        nvtx += 1

        if nvtx>=3:
            ############ Do some calculations #######################
            flight_length.append(magnitude_of_3vec(vtx[0]-vtx[1]))

            nvtx = 0


#print masses
#h1 = plt.hist(masses_inv,200,range=(1.0,1.20),histtype='stepfilled',color='grey')
#h2 = plt.hist(masses_mm,200,range=(1.0,1.20),histtype='stepfilled',color='red',alpha=0.5)
#leg1 = plt.legend(('Invariant mass','Missing mass'),loc='upper right')
#plt.legend([h2],["Missing mass off K"])

#H,xedges,yedges = np.histogram2d(masses_mm,masses_inv,bins=100,range=[[1.10,1.15],[1.10,1.15]])
#H,xedges,yedges = np.histogram2d(masses_mm,masses_inv,bins=100,range=[[1.00,1.50],[1.00,1.50]])
#H,xedges,yedges = np.histogram2d(masses_mm,masses_inv,bins=100,range=[[0.50,1.50],[0.50,1.50]])
# Use this one
#H,xedges,yedges = np.histogram2d(masses_mm,masses_inv,bins=100,range=[[0.90,1.40],[0.90,1.40]])
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

#plt.imshow(H,extent=extent,interpolation='nearest',origin='lower',cmap=plt.cm.Spectral)
#plt.imshow(H,extent=extent,interpolation='nearest',origin='lower',cmap=plt.cm.seismic)
#ax = plt.axes()
#ax.set_xlabel("Invariant mass of $X^+ X^-$ system",size=20)
#ax.set_ylabel("Missing mass off $K^+$",size=20)
#plt.imshow(H,extent=extent,interpolation='nearest',origin='lower',cmap=plt.cm.coolwarm,axes=ax)
#plt.colorbar()
#test

plt.hist(flight_length,100,range=(0.0,50.0),histtype='stepfilled',alpha=1.0)
plt.savefig("SM_fligh_len_0-1.png")
#plt.savefig("SM_fligh_len_0-2.png")

#plt.savefig("SM_Kp_p_pim.png")
#plt.savefig("SM_Kp_Kp_em_assumption.png")
#plt.savefig("Kp_Kp_em_correct_PID.png")

plt.show()

