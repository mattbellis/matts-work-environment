#!/usr/bin/env python

import sys
from math import *

import numpy as np

from StringIO import StringIO

mass_e = 0.000511
mass_pi = 0.139570
mass_k = 0.493677
mass_p = 0.938272
mass_mu = 0.105658

target = np.array([mass_p,0.0,0.0,0.0])
target = np.array([mass_p,0.0,0.0,0.0])

################################################################################
# Calculate the magnitude of a three-vector
################################################################################
def magnitude_of_3vec(v3):

    magnitude = np.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

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

    '''
    if mass_squared>0:
        return np.sqrt(mass_squared)
    else:
        return -np.sqrt(abs(mass_squared))
    '''
    return np.sqrt(mass_squared)

################################################################################
# Calculate the mass of a particle using Special Relativity
################################################################################
def mass2_from_special_relativity(v4):

    pmag2 = v4[1]**2 + v4[2]**2 + v4[3]**2

    mass_squared = v4[0]**2 - pmag2

    return mass_squared

################################################################################

################################################################################
# Read in the files from McCracken
################################################################################
def read_mmcc_inputfiles(infile_names,max_events=1e12):

    e0 = np.array([])
    e1 = np.array([])
    e2 = np.array([])

    p0 = []
    p1 = []
    p2 = []
    for i in xrange(5):
        p0.append(np.array([]))
        p1.append(np.array([]))
        p2.append(np.array([]))

    vtx0 = []
    vtx1 = []
    vtx2 = []
    vtx3 = []
    for i in xrange(3):
        vtx0.append(np.array([]))
        vtx1.append(np.array([]))
        vtx2.append(np.array([]))
        vtx3.append(np.array([]))

    beam_e = 0.0
    beam = []
    for i in xrange(4):
        beam.append(np.array([]))
    
    event = [None,None,None,None,None,None,None,None]

    count = 0

    for infile_name in infile_names:
        print infile_name
        infile = open(infile_name)

        line = ' '
        while len(line)!=0:
            #for line in infile:
            line = infile.readline()
            #event[0] = line.split()
            #event[0] = np.genfromtxt(StringIO(line),dtype=(float))
            event[0] = np.loadtxt(StringIO(line),dtype=(float))
            good_event = 'nan' not in event[0] and 'inf' not in event[0]
            #print line
            #print event[0]
            if len(event[0])==2:
                for i in xrange(1,8):
                    line = infile.readline()
                    event[i] = np.genfromtxt(StringIO(line),dtype=(float))
                    #event[i] = infile.readline().split()
                    good_event *= 'nan' not in event[i] and 'inf' not in event[i]
                    

                #print good_event
                if good_event:

                    beam_e = event[0][1]
                    beam[0] = np.append(beam[0],beam_e)
                    beam[1] = np.append(beam[1],0.0)
                    beam[2] = np.append(beam[2],0.0)
                    beam[3] = np.append(beam[3],beam_e)

                    for j in xrange(0,5):
                        p0[j] = np.append(p0[j],float(event[1][j]))
                        p1[j] = np.append(p1[j],float(event[2][j]))
                        p2[j] = np.append(p2[j],float(event[3][j]))

                    for j in xrange(0,3):
                        vtx0[j] = np.append(vtx0[j],float(event[4][j]))
                        vtx1[j] = np.append(vtx1[j],float(event[5][j]))
                        vtx2[j] = np.append(vtx2[j],float(event[6][j]))
                        vtx3[j] = np.append(vtx3[j],float(event[7][j]))

                    count +=1 
                    if count%1000==0:
                        print count
                    if count > max_events:
                        break

    return beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3

