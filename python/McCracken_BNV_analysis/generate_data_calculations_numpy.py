#!/usr/bin/env python

import sys
from math import *

import numpy as np
import matplotlib.pyplot as plt

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

    print beam

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

    ############################################################################

    infile_names = args.input_file_name
    print infile_names
    #outfile_name = "calculated_data_files/%s_hyp_%s.csv" % (infile_name.split('/')[-1].split('.')[0],args.hypothesis)
    outfile_name = "calculated_data_files/%s_hyp_%s.csv" % (infile_names[0][0].split('/')[-1].split('.')[0],args.hypothesis)
    print outfile_name
    outfile = open(outfile_name,"w+")

    ############################################################################
    for infile_name in infile_names[0]:
        infile = open(infile_name)

        for line in infile:
            vals = line.split()

            if len(vals)==2:

                beam_e = float(vals[1])
                beam[0] = np.append(beam[0],beam_e)
                beam[1] = np.append(beam[1],0.0)
                beam[2] = np.append(beam[2],0.0)
                beam[3] = np.append(beam[3],beam_e)

                n=0
                nvtx=0

            elif len(vals)==5:

                if n==0:
                    for j in xrange(0,5):
                        p0[j] = np.append(p0[j],float(vals[j]))
                elif n==1:
                    for j in xrange(0,5):
                        p1[j] = np.append(p1[j],float(vals[j]))
                elif n==2:
                    for j in xrange(0,5):
                        p2[j] = np.append(p2[j],float(vals[j]))

                n += 1

            elif len(vals)==3:

                if nvtx==0:
                    for j in xrange(0,3):
                        vtx0[j] = np.append(vtx0[j],float(vals[j]))
                elif nvtx==1:
                    for j in xrange(0,3):
                        vtx1[j] = np.append(vtx1[j],float(vals[j]))
                elif nvtx==2:
                    for j in xrange(0,3):
                        vtx2[j] = np.append(vtx2[j],float(vals[j]))
                elif nvtx==3:
                    for j in xrange(0,3):
                        vtx3[j] = np.append(vtx3[j],float(vals[j]))

                nvtx += 1

        ############################################################################
        # Finished reading in the data.
        ############################################################################
        print "Beam stuff: " 
        print len(beam[0])
        print beam[0]
        beam_plus_target = [beam[0]+mass_p,beam[1],beam[2],beam[3]]
        print beam_plus_target
        print "end"
        print p0
        ############ Do some calculations #######################

        output += ""
        inv_mass = []
        missing_mass = []
        missing_mass_off_kaon = []
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
            #'''
            p4 = [ beam_plus_target[0] - e0 - e1 - e2,
                   beam_plus_target[1] - p0[0] - p1[0] - p2[0],
                   beam_plus_target[2] - p0[1] - p1[1] - p2[1],
                   beam_plus_target[3] - p0[2] - p1[2] - p2[2]]
            #'''

            #print p4
            missing_mass.append(mass2_from_special_relativity(p4))

            # Missing mass off kaon
            #'''
            p4 = [ beam_plus_target[0] - e0,
                   beam_plus_target[1] - p0[0],
                   beam_plus_target[2] - p0[1],
                   beam_plus_target[3] - p0[2]]
            #'''

            #print p4
            missing_mass_off_kaon.append(mass_from_special_relativity(p4))

            # Lambda mass
            #'''
            p4 = [ e1 + e2,
                   p1[0] + p2[0],
                   p1[1] + p2[1],
                   p1[2] + p2[2]]
            #'''

            #print p4
            inv_mass.append(mass_from_special_relativity(p4))

        output += ""
        for m0,mm0,mk0,m1,mm1,mk1 in zip(inv_mass[0],missing_mass[0],missing_mass_off_kaon[0],inv_mass[1],missing_mass[1],missing_mass_off_kaon[1]):
            output = "%f %f %f %f %f %f\n" % (m0,mm0,mk0,m1,mm1,mk1)
            outfile.write(output)

    ############################################################################
    # Finished reading and writing all the data.
    ############################################################################
    outfile.close()




            #plt.hist(m,bins=100,range=(1.0,1.5))
            #plt.show()

                #p[i] = np.insert(v3[i],[0],energy)

    '''
                if nvtx>=3:

                    ############ Do some calculations #######################
                    print beam_plus_target
                    for j in xrange(2):

                        #### Go through the two mass hypothesis.
                        for i,mass in enumerate(masses[j]):

                            pmag = magnitude_of_3vec(v3[i])
                            energy = sqrt(mass*mass + pmag*pmag)
                            p[i] = np.insert(v3[i],[0],energy)

                        ##### Missing mass
                        p4 = beam_plus_target - p[0] - p[1] - p[2]
                        output += "%f," % (mass_from_special_relativity(p4))

                        # Assuming first + particle is a kaon
                        if j==0:
                            p4 = beam_plus_target - p[0]
                            output += "%f," % (mass_from_special_relativity(p4))

                            p4 = p[1]+p[2]
                            output += "%f," % (mass_from_special_relativity(p4))

                        # Assuming second + particle is a kaon
                        elif j==1:
                            p4 = beam_plus_target - p[1]
                            output += "%f," % (mass_from_special_relativity(p4))

                            p4 = p[0]+p[2]
                            output += "%f," % (mass_from_special_relativity(p4))

                    n=0
                    count +=1 
                    if count%10000==0:
                        print count
                    if count > 10000:
                        break

            elif len(vals)==3:

                x = [float(vals[0]),float(vals[1]),float(vals[2])]
                v3vtx = np.array(x)

                vtx[nvtx] = v3vtx

                nvtx += 1

                if nvtx>=4:
                    ############ Do some calculations #######################
                    output += "%f," % (magnitude_of_3vec(vtx[0]-vtx[2]))
                    output += "%f\n" % (magnitude_of_3vec(vtx[1]-vtx[3]))

                    nvtx = 0
                    outfile.write(output)
                    output = ""


        outfile.close()
    '''

################################################################################
################################################################################
if __name__ == "__main__":
    main()

