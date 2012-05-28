#!/usr/bin/env python

import sys
from math import *

import numpy as np
import matplotlib.pyplot as plt

from StringIO import StringIO

from analysis_utilities import *

import argparse

################################################################################
# main
################################################################################
def main():

    ############################################################################
    # Parse the arguments
    ############################################################################
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file_name', action='append', nargs='*',
            help='Input file name')
    parser.add_argument('--hypothesis',dest='hypothesis',type=str, default='kppi', help='Hypothesis of the ++- particle topology.')

    args = parser.parse_args()

    ############################################################################

    if len(args.input_file_name)==0:
        print "Must pass in an input file name!"
        parser.print_help()

    ############################################################################

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

    #print beam

    n = 0
    masses = []
    masses_inv = []
    masses_mm = []
    flight_length = []
    vtx = [0.0,0.0,0.0,0.0]
    count = 0
    nvtx = 0
    output = ""
    v3 = [0.0, 0.0, 0.0]

    if args.hypothesis == 'kppi':
        masses = [[mass_k,mass_p,mass_pi],
                  [mass_p,mass_k,mass_pi]]

    elif args.hypothesis == 'kek':
        masses = [[mass_k,mass_e,mass_k],
                  [mass_e,mass_k,mass_k]]

    elif args.hypothesis == 'kepi':
        masses = [[mass_k,mass_e,mass_pi],
                  [mass_e,mass_k,mass_pi]]

    elif args.hypothesis == 'kkmu':
        masses = [[mass_k,mass_k,mass_mu],
                  [mass_pi,mass_k,mass_mu]]

    ############################################################################

    infile_names = args.input_file_name
    #print infile_names
    #outfile_name = "calculated_data_files/%s_hyp_%s.csv" % (infile_name.split('/')[-1].split('.')[0],args.hypothesis)
    outfile_name = "calculated_data_files/%s_hyp_%s" % (infile_names[0][0].split('/')[-1].split('.')[0],args.hypothesis)
    #print outfile_name
    #outfile = open(outfile_name,"w+")

    ############################################################################
    beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3 = read_mmcc_inputfiles(infile_names[0],1000)

    '''
    event = [None,None,None,None,None,None,None,None]

    for infile_name in infile_names[0]:
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
                    if count > 100000:
                        break

        '''
    ############################################################################
    # Finished reading in the data.
    ############################################################################
    #print "Beam stuff: " 
    #print len(beam[0])
    #print beam[0]
    beam_plus_target = [beam[0]+mass_p,beam[1],beam[2],beam[3]]
    #print beam_plus_target
    #print "end"
    #print p0
    ############ Do some calculations #######################

    output += ""
    inv_mass = []
    missing_mass = []
    missing_mass_off_kaon = []
    flightlen = []
    p4 = None
    for j in xrange(2):

        #### Go through the two mass hypothesis.
        #for i,mass in enumerate(masses[j]):

        #for pa0,pa1,pa2 in zip(p0,p1,p2):
        pmag0 = magnitude_of_3vec(p0[0:3])
        e0 = np.sqrt(masses[j][0]**2 + pmag0**2)

        pmag1 = magnitude_of_3vec(p1[0:3])
        e1 = np.sqrt(masses[j][1]**2 + pmag1**2)

        pmag2 = magnitude_of_3vec(p2[0:3])
        e2 = np.sqrt(masses[j][2]**2 + pmag2**2)

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
        else:
            p4 = [ e0 + e2,
                   p0[0] + p2[0],
                   p0[1] + p2[1],
                   p0[2] + p2[2]]

        #print p4
        inv_mass.append(mass_from_special_relativity(p4))

        vec = [vtx0[0]-vtx3[0],vtx0[1]-vtx3[1],vtx0[2]-vtx3[2]]
        flightlen.append(magnitude_of_3vec(vec))
        vec = [vtx1[0]-vtx2[0],vtx1[1]-vtx2[1],vtx1[2]-vtx2[2]]
        flightlen.append(magnitude_of_3vec(vec))

    output += ""
    '''
    for m0,m1,mm0,mm1,mk0,mk1,fl0,fl1 in zip(inv_mass[0],missing_mass[0],missing_mass_off_kaon[0],flightlen[0],inv_mass[1],missing_mass[1],missing_mass_off_kaon[1],flightlen[1]):
        output = "%f %f %f %f %f %f %f %f\n" % (m0,mm0,mk0,fl0,m1,mm1,mk1,fl1)
        outfile.write(output)
    '''
    outarrays = [missing_mass[0],\
                missing_mass[1],\
                inv_mass[0],\
                inv_mass[1],\
                missing_mass_off_kaon[0],\
                missing_mass_off_kaon[1],\
                flightlen[0],\
                flightlen[1]]

    #np.savetxt(outfile_name,outarrays)
    np.save(outfile_name,outarrays)

############################################################################
# Finished reading and writing all the data.
############################################################################
#outfile.close()


################################################################################
################################################################################
if __name__ == "__main__":
    main()

