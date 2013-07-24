import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime,timedelta

import scipy.integrate as integrate

import parameters 
from cogent_utilities import *
from fitting_utilities import *
from lichen.plotting_utilities import *
import lichen.pdfs as pdfs

import lichen.iminuit_fitting_utilities as fitutils

import lichen.lichen as lch

import iminuit as minuit

import argparse

import math

pi = np.pi
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

np.random.seed(200)

yearly_mod = 2*pi/365.0


################################################################################
# Read in the CoGeNT data
################################################################################
def main():

    ############################################################################
    # Parse the command lines.
    ############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', dest='fit', type=int,\
            default=0, help='Which fit to perform (0,1,2)')
    parser.add_argument('--verbose', dest='verbose', action='store_true',\
            default=False, help='Verbose output')
    parser.add_argument('--sigma_n', dest='sigma_n', type=float,\
            default=None, help='Value of sigma_n (cross section of DM-nucleon interaction).')
    parser.add_argument('--turn-off-eff', dest='turn_off_eff', action='store_true',\
            default=False, help='Turn off the efficiency.')
    parser.add_argument('--contours', dest='contours', action='store_true',\
            default=False, help='Calculate and plot the contours.')
    parser.add_argument('--batch', dest='batch', action='store_true',\
            default=False, help='Run in batch mode (exit on completion).')

    args = parser.parse_args()

    ############################################################################

    '''
    if args.help:
        parser.print_help()
        exit(-1)
    '''

    ############################################################################
    # Read in the data
    ############################################################################
    infile_name = 'data/LE.txt'
    tdays,energies,rise_times = get_3yr_cogent_data(infile_name,first_event=first_event,calibration=0)
    print tdays
    print energies
    print rise_times

    print energies
    if args.verbose:
        print_data(energies,tdays,rise_times)

    #data = [energies.copy(),tdays.copy()]
    #print "data before range cuts: ",len(data[0]),len(data[1])

    # 3yr data
    data = [energies.copy(),tdays.copy(),rise_times]
    print "data before range cuts: ",len(data[0]),len(data[1]),len(data[2])
    #exit()


    ############################################################################
    # Declare the ranges.
    ############################################################################
    ranges,subranges,nbins = parameters.fitting_parameters(args.fit)
    
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n

    # Cut events out that fall outside the range.
    data = cut_events_outside_range(data,ranges)
    data = cut_events_outside_subrange(data,subranges[1],data_index=1)

    if args.verbose:
        print_data(energies,tdays)

    print "data after  range cuts: ",len(data[0]),len(data[1])

    nevents = float(len(data[0]))

    # Plot rise times vs. energies
    plt.figure()
    plt.plot(data[0],data[2],'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)

    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    params_file = 'rt_parameters.txt'
    rt_parameters = np.loadtxt(params_file)
    print rt_parameters[0]
    print rt_parameters[1]
    print rt_parameters[5]
    print rt_parameters[6]
    print rt_parameters[7]

    fmu = []
    fsig = []
    fn = []
    smu = []
    ssig = []
    sn = []
    npts = []
    print "Figuring out the parameters for each data point...."
    for e in data[0]:
        # Loop over the energy slices
        found = False
        for i,elo in enumerate(rt_parameters[0]):
            #print i,elo
            if (e>=elo-0.02500001 and e<elo+0.02500001) or e<0.600:
                fmu.append(rt_parameters[1][i])
                fsig.append(rt_parameters[2][i])
                fn.append(rt_parameters[3][i])
                smu.append(rt_parameters[4][i])
                ssig.append(rt_parameters[5][i])
                sn.append(rt_parameters[6][i])
                npts.append(rt_parameters[7][i])
                found = True
                break

        if found==False:
            print e

    print "Figured out the parameters for each data point!"
    print len(data[0]),len(fmu)
    nevents = float(len(data[0]))

    fweights = []
    sweights = []
    for i,rt in enumerate(data[2]):
        # Loop over the energy slices
        #print i,e,fmu[i],fsig[i],fn[i]
        w = fn[i]*pdfs.lognormal(rt,fmu[i],fsig[i],ranges[2][0],ranges[2][1])
        #w /= nevents
        fweights.append(w)
        w = sn[i]*pdfs.lognormal(rt,smu[i],ssig[i],ranges[2][0],ranges[2][1])
        #w /= nevents
        sweights.append(w)

    plt.figure()
    lch.hist_err(data[2],bins=nbins[2],range=(ranges[2][0],ranges[2][1]))
    plt.xlim(ranges[2][0],ranges[2][1])

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(data[2],bins=nbins[2],range=(ranges[2][0],ranges[2][1]),weights=fweights)
    plt.xlim(ranges[2][0],ranges[2][1])

    plt.subplot(1,2,2)
    lch.hist_err(data[2],bins=nbins[2],range=(ranges[2][0],ranges[2][1]),weights=sweights)
    plt.xlim(ranges[2][0],ranges[2][1])


    plt.figure()
    lch.hist_err(data[0],bins=nbins[0],range=(ranges[0][0],ranges[0][1]))
    plt.xlim(ranges[0][0],ranges[0][1])

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(data[0],bins=nbins[0],range=(ranges[0][0],ranges[0][1]),weights=fweights)
    plt.xlim(ranges[0][0],ranges[0][1])
    plt.subplot(1,2,2)
    lch.hist_err(data[0],bins=nbins[0],range=(ranges[0][0],ranges[0][1]),weights=sweights)
    plt.xlim(ranges[0][0],ranges[0][1])

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(fweights,bins=50)
    plt.subplot(1,2,2)
    lch.hist_err(sweights,bins=50)

    #plt.figure()
    #lch.hist_2D(data[0],fweights,xbins=nbins[0],ybins=100,xrange=(ranges[0][0],ranges[0][1]),yrange=(0,0.0001))

    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
