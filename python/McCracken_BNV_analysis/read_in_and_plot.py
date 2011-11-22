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

p = [0.0, 0.0, 0.0, 0.0, 0.0]
n = 0
masses = []
count = 0
for line in infile:
    vals = line.split()
    if len(vals)==5:
        #print vals

        vec3 = [float(vals[0]),float(vals[1]),float(vals[2])]

        energy = 0
        pmag = magnitude_of_3vec(vec3)
        if n==0:
            mass = mass_k
        elif n==1:
            mass = mass_pi
        elif n==2:
            mass = mass_e
        energy = sqrt(mass*mass + pmag*pmag)

        p[n] = [energy] + vec3
        #print p[n]

        n += 1

        if n>=3:
            ############ Do some calculations #######################
            #mass.append(mass_from_special_relativity(p[3],p[4]))
            p4 = map(sum, zip(p[1],p[2]))
            #p4 = p[0]
            #print mass_from_special_relativity(p4)
            masses.append(mass_from_special_relativity(p4))

            n=0
            count +=1 
            if count > 100000:
                break

        #print n
        #print p

#print masses
plt.hist(masses,100,range=(0.0,2.0),histtype='stepfilled')
plt.show()

