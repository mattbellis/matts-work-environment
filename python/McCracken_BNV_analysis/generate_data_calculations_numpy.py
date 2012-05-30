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
    beta0 = []
    beta1 = []
    beta2 = []
    flight_length = []
    vtx = [0.0,0.0,0.0,0.0]
    count = 0
    nvtx = 0
    v3 = [0.0, 0.0, 0.0]

    if args.hypothesis == 'kke':
        masses = [[mass_k,mass_k,mass_e],
                  [mass_k,mass_k,mass_e]]

    elif args.hypothesis == 'kek':
        masses = [[mass_k,mass_e,mass_k],
                  [mass_e,mass_k,mass_k]]

    elif args.hypothesis == 'kkmu':
        masses = [[mass_k,mass_k,mass_mu],
                  [mass_k,mass_k,mass_mu]]

    elif args.hypothesis == 'kmuk':
        masses = [[mass_k,mass_mu,mass_k],
                  [mass_mu,mass_k,mass_k]]

    elif args.hypothesis == 'kpie':
        masses = [[mass_k,mass_pi,mass_e],
                  [mass_pi,mass_k,mass_e]]

    elif args.hypothesis == 'kepi':
        masses = [[mass_k,mass_e,mass_pi],
                  [mass_e,mass_k,mass_pi]]

    elif args.hypothesis == 'kpimu':
        masses = [[mass_k,mass_pi,mass_mu],
                  [mass_pi,mass_k,mass_mu]]

    elif args.hypothesis == 'kmupi':
        masses = [[mass_k,mass_mu,mass_pi],
                  [mass_mu,mass_k,mass_pi]]

    elif args.hypothesis == 'kppi':
        masses = [[mass_k,mass_p,mass_pi],
                  [mass_p,mass_k,mass_pi]]

    ############################################################################
    ############################################################################

    infile_names = args.input_file_name
    outfile_name = "calculated_data_files/%s_hyp_%s" % (infile_names[0][0].split('/')[-1].split('.')[0],args.hypothesis)
    
    #beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3 = read_mmcc_inputfiles(infile_names[0])
    beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3 = read_mmcc_inputfiles_numpy(infile_names[0])

    #print beam
    #exit(-1)

    ############################################################################
    # Finished reading in the data.
    ############################################################################
    beam_plus_target = [beam[0]+mass_p,beam[1],beam[2],beam[3]]
    
    ############ Do some calculations #######################

    inv_mass = []
    missing_mass = []
    missing_mass_off_kaon = []
    #beta = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
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

    ############################################################################
    # Write out the data to a numpy file.
    ############################################################################
    outarrays = [missing_mass[0],\
                missing_mass[1],\
                inv_mass[0],\
                inv_mass[1],\
                missing_mass_off_kaon[0],\
                missing_mass_off_kaon[1],\
                flightlen[0],\
                flightlen[1],
                beta0[0],beta0[1],
                beta1[0],beta1[1],
                beta2[0],beta2[1],
                lambda_beta[0],lambda_beta[1],
                meas_beta[0],meas_beta[1],meas_beta[2]
                ]

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

