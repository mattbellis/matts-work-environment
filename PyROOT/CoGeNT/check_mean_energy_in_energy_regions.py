#!/usr/bin/env python

import numpy as np

from cogent_utilities import *

import argparse
from datetime import datetime,timedelta

################################################################################
################################################################################
def main():
    
    ############################################################################
    # Parse the command lines.
    ############################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file_name', type=str, default=None,
            help='Input file name')
    parser.add_argument('--e-lo', dest='e_lo', type=float, default=0.5,
            help='Set the lower limit for the energy range to use.')
    parser.add_argument('--e-hi', dest='e_hi', type=float, default=3.2,
            help='Set the upper limit for the energy range to use.')
    parser.add_argument('--calib', dest='calib', type=int,
            default=0, help='Which calibration to use (0,1,2)')


    parser.add_argument('--myhelp', dest='help', action='store_true',
            default=False, help='Print help options.')

    args = parser.parse_args()

    ############################################################################
    
    if args.help:
        parser.print_help()
        exit(-1)

    if args.input_file_name is None:
        print "Must pass in an input file name!"
        parser.print_help()

    infile_name = args.input_file_name
    ############################################################################

    ############################################################################
    # Put some configuration stuff here.
    ############################################################################
    # Time of the first event in seconds. We will
    # need this for converting the times in the input file.
    first_event = 2750361.2
    # First day of data recording.
    start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

    # Max day for plotting
    tmax = 480;
    # When plotting the time, use this for binning.
    tbins = 16;
    t_bin_width = tmax/tbins

    # Energy fitting range.
    lo_energy = args.e_lo
    hi_energy = args.e_hi


    infile = open(infile_name)

    time_stamp = []
    energy = []

    for line in infile:

        vals = line.split()


        # Make sure there are at least two numbers on a line.
        if len(vals)==2:

            t_sec = float(vals[0])
            amplitude = float(vals[1])

            # Convert the amplitude to an energy using a particular calibration.
            energy.append(amp_to_energy(amplitude,args.calib))

            # Convert the time in seconds to a day.
            time_stamp.append((t_sec-first_event)/(24.0*3600.0) + 1.0)

    #print energy
    #print time_stamp

    month = 0
    energies = []
    xpts = []
    ypts = []
    print "Month  Mean  Std.Dev."
    for e,t in zip(energy,time_stamp):

        if t>30*(month+1):
            month += 1
            #print energies
            print "%2d   %6.3f   %4.3f" % (month, np.mean(energies),np.std(energies))
            energies = []

        if t>month*30 and t<(month+1)*30:
            if e>=lo_energy and e<hi_energy:
                energies.append(e)



################################################################################
################################################################################
if __name__ == "__main__":
    main()




