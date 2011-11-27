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
mass_pi = 0.139
mass_k = 0.494
mass_p = 0.938

target = np.array([mass_p,0.0,0.0,0.0])

p = [0.0, 0.0, 0.0, 0.0, 0.0]
n = 0
masses = []
masses_inv = []
masses_mm = []
count = 0
beam = None
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
            mass = mass_k
        elif n==2:
            #mass = mass_k
            mass = mass_e
        energy = sqrt(mass*mass + pmag*pmag)

        p[n] = np.insert(v3,[0],energy)
        #print p[n]
        #print p[n]

        n += 1

        if n>=3:
            ############ Do some calculations #######################
            p4 = beam + target - p[0]
            masses_mm.append(mass_from_special_relativity(p4))

            p4 = p[1]+p[2]
            masses_inv.append(mass_from_special_relativity(p4))

            n=0
            count +=1 
            if count > 10000000:
                break

        #print n
        #print p

#print masses
#plt.hist(masses_inv,200,range=(1.0,1.2),histtype='stepfilled')
#plt.hist(masses_mm,200,range=(1.0,1.2),histtype='stepfilled',alpha=0.5)

H,xedges,yedges = np.histogram2d(masses_mm,masses_inv,bins=100,range=[[1.0,1.2],[1.0,1.2]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
plt.imshow(H,extent=extent,interpolation='nearest',origin='lower')
plt.colorbar()

plt.show()

