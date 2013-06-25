#!/usr/bin/env python

import sys
from math import *
import os

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
    parser.add_argument('--out-subdir',dest='out_subdir',type=str, default=None, help='Subdirectory for output files.')

    args = parser.parse_args()

    ############################################################################

    if len(args.input_file_name)==0:
        print "Must pass in an input file name!"
        parser.print_help()

    ############################################################################

    masses = get_mass_hypothesis_permutations(args.hypothesis)
    
    ############################################################################

    infile_names = args.input_file_name
    outfile_name = "calculated_data_files/%s_hyp_%s" % (infile_names[0][0].split('/')[-1].split('.')[0],args.hypothesis)
    if args.out_subdir!=None:
        dir_name = "./calculated_data_files/%s" % (args.out_subdir)
        if not os.access(dir_name,os.W_OK ):
            os.mkdir(dir_name,0744)
        outfile_name = "calculated_data_files/%s/%s_hyp_%s" % (args.out_subdir,infile_names[0][0].split('/')[-1].split('.')[0],args.hypothesis)
        print outfile_name
    
    beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3 = read_mmcc_inputfiles_numpy(infile_names[0])

    ############################################################################
    # Finished reading in the data.
    ############################################################################
    missing_mass,inv_mass,missing_mass_off_kaon,\
                        flightlen,beta0,beta1,beta2,lambda_beta,meas_beta = \
    process_data(masses,beam,e0,e1,e2,p0,p1,p2,vtx0,vtx1,vtx2,vtx3)

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

    np.save(outfile_name,outarrays)

############################################################################
# Finished reading and writing all the data.
############################################################################


################################################################################
################################################################################
if __name__ == "__main__":
    main()

