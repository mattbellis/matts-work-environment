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
mass_L = 1.115683

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
    #print v4[0][mass_squared!=mass_squared]
    #print pmag[mass_squared!=mass_squared]
    #print mass_squared[mass_squared!=mass_squared]
    mass_squared[mass_squared!=mass_squared] = 999999
    #print mass_squared[mass_squared!=mass_squared]
    #print mass_squared[mass_squared<=0.0]
    #print len(mass_squared[mass_squared<=0.0])

    mass_squared_sqrt = np.zeros(len(mass_squared))
    mass_squared_sqrt[mass_squared<0.0] = np.sqrt(-mass_squared[mass_squared<0.0])
    mass_squared_sqrt[mass_squared>=0.0] = np.sqrt(mass_squared[mass_squared>=0.0])

    #return np.sqrt(mass_squared)
    #print mass_squared_sqrt
    return mass_squared_sqrt

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
            line = infile.readline()
            event[0] = np.loadtxt(StringIO(line),dtype=(float))
            good_event = 'nan' not in event[0] and 'inf' not in event[0]
            if len(event[0])==2:
                for i in xrange(1,8):
                    line = infile.readline()
                    event[i] = np.genfromtxt(StringIO(line),dtype=(float))
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

################################################################################
# Read in the files from McCracken
################################################################################
def read_mmcc_inputfiles_numpy(infile_names,max_events=1e12):

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

        # Read in the entire content of the file and split it.
        # Save it as an numpy array.
        # Also turn it into floats.
        content = np.array(infile.read().split()).astype('float')
        nentries = len(content)
        nevents = nentries/29
        print "# of entries: %d" % (nentries)
        print "# of events: %d" % (nevents)

        # Build up an index array to pull out the elements.
        index = np.arange(0,nentries,29)

        beam[0] = np.append(beam[0],content[index+1])
        beam[1] = np.append(beam[1],np.zeros(nevents))
        beam[2] = np.append(beam[2],np.zeros(nevents))
        beam[3] = np.append(beam[3],content[index+1])

        for j in xrange(5):
            p0[j] = np.append(p0[j],content[index+(2+j)])
            p1[j] = np.append(p1[j],content[index+(7+j)])
            p2[j] = np.append(p2[j],content[index+(12+j)])

        for j in xrange(3):
            vtx0[j] = np.append(vtx0[j],content[index+(17+j)])
            vtx1[j] = np.append(vtx1[j],content[index+(20+j)])
            vtx2[j] = np.append(vtx2[j],content[index+(23+j)])
            vtx3[j] = np.append(vtx3[j],content[index+(26+j)])

    return beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3

################################################################################
# Process these data.
################################################################################
def get_mass_hypothesis_permutations(hypothesis):

    masses = None 

    if hypothesis == 'kke':
        masses = [[mass_k,mass_k,mass_e],
                  [mass_k,mass_k,mass_e]]

    elif hypothesis == 'kek':
        masses = [[mass_k,mass_e,mass_k],
                  [mass_e,mass_k,mass_k]]

    elif hypothesis == 'kkmu':
        masses = [[mass_k,mass_k,mass_mu],
                  [mass_k,mass_k,mass_mu]]

    elif hypothesis == 'kmuk':
        masses = [[mass_k,mass_mu,mass_k],
                  [mass_mu,mass_k,mass_k]]

    elif hypothesis == 'kpie':
        masses = [[mass_k,mass_pi,mass_e],
                  [mass_pi,mass_k,mass_e]]

    elif hypothesis == 'kepi':
        masses = [[mass_k,mass_e,mass_pi],
                  [mass_e,mass_k,mass_pi]]

    elif hypothesis == 'kpimu':
        masses = [[mass_k,mass_pi,mass_mu],
                  [mass_pi,mass_k,mass_mu]]

    elif hypothesis == 'kmupi':
        masses = [[mass_k,mass_mu,mass_pi],
                  [mass_mu,mass_k,mass_pi]]

    elif hypothesis == 'kppi':
        masses = [[mass_k,mass_p,mass_pi],
                  [mass_p,mass_k,mass_pi]]

    return masses


################################################################################
# Process these data.
################################################################################
def process_data(masses,beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3):

    beam_plus_target = [beam[0]+mass_p,beam[1],beam[2],beam[3]]

    ############ Do some calculations #######################
    beta0 = []
    beta1 = []
    beta2 = []
    flight_length = []
    vtx = [0.0,0.0,0.0,0.0]
    count = 0
    nvtx = 0
    v3 = [0.0, 0.0, 0.0]

    inv_mass = []
    missing_mass = []
    missing_mass_off_kaon = []
    lambda_beta = []
    meas_beta = [np.array([]),np.array([]),np.array([])]
    flightlen = []
    p4 = None
    for j in xrange(2):

        if j==0:
            meas_beta[0] = p0[3]/(p0[4]*29.97)
            meas_beta[1] = p1[3]/(p1[4]*29.97)
            meas_beta[2] = p2[3]/(p2[4]*29.97)

        #### Go through the two mass hypothesis.
        pmag0 = magnitude_of_3vec(p0[0:3])
        e0 = np.sqrt(masses[j][0]**2 + pmag0**2)
        beta0.append(pmag0/e0)

        pmag1 = magnitude_of_3vec(p1[0:3])
        e1 = np.sqrt(masses[j][1]**2 + pmag1**2)
        beta1.append(pmag1/e1)

        pmag2 = magnitude_of_3vec(p2[0:3])
        e2 = np.sqrt(masses[j][2]**2 + pmag2**2)
        beta2.append(pmag2/e2)

        # Missing mass
        p4 = [ beam_plus_target[0] - e0 - e1 - e2,
               beam_plus_target[1] - p0[0] - p1[0] - p2[0],
               beam_plus_target[2] - p0[1] - p1[1] - p2[1],
               beam_plus_target[3] - p0[2] - p1[2] - p2[2]]

        #print p4
        missing_mass.append(mass2_from_special_relativity(p4))

        # Missing mass off kaon
        if j==0:
            p4 = [ beam_plus_target[0] - e0,
                   beam_plus_target[1] - p0[0],
                   beam_plus_target[2] - p0[1],
                   beam_plus_target[3] - p0[2]]
        else:
            p4 = [ beam_plus_target[0] - e1,
                   beam_plus_target[1] - p1[0],
                   beam_plus_target[2] - p1[1],
                   beam_plus_target[3] - p1[2]]

        #print p4
        missing_mass_off_kaon.append(mass_from_special_relativity(p4))

        # Lambda mass
        if j==0:
            p4 = [ e1 + e2,
                   p1[0] + p2[0],
                   p1[1] + p2[1],
                   p1[2] + p2[2]]

            pmag = magnitude_of_3vec(p1[0:3]+p2[0:3])
            lambda_beta.append(pmag/(e1+e2))

        else:
            p4 = [ e0 + e2,
                   p0[0] + p2[0],
                   p0[1] + p2[1],
                   p0[2] + p2[2]]

            pmag = magnitude_of_3vec(p0[0:3]+p2[0:3])
            lambda_beta.append(pmag/(e0+e2))

        #print p4
        inv_mass.append(mass_from_special_relativity(p4))

        vec = [vtx0[0]-vtx3[0],vtx0[1]-vtx3[1],vtx0[2]-vtx3[2]]
        flightlen.append(magnitude_of_3vec(vec))
        vec = [vtx1[0]-vtx2[0],vtx1[1]-vtx2[1],vtx1[2]-vtx2[2]]
        flightlen.append(magnitude_of_3vec(vec))

    ############################################################################
    # Finished with the calculations.
    ############################################################################

    return missing_mass,inv_mass,missing_mass_off_kaon,\
            flightlen,beta0,beta1,beta2,lambda_beta,meas_beta

