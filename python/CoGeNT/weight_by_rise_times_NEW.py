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

from scipy.optimize import curve_fit

################################################################################
#def func(x, a, b, c, d):
    #return a + b*x + c*x*x + d*x*x*x

def func(x, a, b, c):
    return a*np.exp(-b*x) + c

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
    #infile_name = 'data/HE.txt'
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
    '''
    plt.figure()
    plt.plot(data[0],data[2],'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)
    '''

    #plt.show()
    #exit()
    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    print "Precalculating the fast and slow rise time probabilities........"
    ############################################################################
    # Parameters for the exponential form for the narrow fast peak.
    mu0 =  [1.016749,0.786867,-1.203125]
    sigma0 =  [2.372789,1.140669,0.262251]
    # The entries for the relationship between the broad and narrow peak.
    fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
    fast_sigma_rel_k = [-3.667547,0.000256,-0.364826]
    fast_num_rel_k =  [-2.831665,0.023649,1.144240]
    rt_fast = rise_time_prob_fast_exp_dist(data[2],data[0],mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

    # Parameters for the exponential form for the slow peak.
    mu = [0.945067,0.646431,0.353891]
    sigma =  [11.765617,94.854276,0.513464]
    rt_slow = rise_time_prob_exp_progression(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])
    ############################################################################

    rt_fast /= (ranges[0][1]-ranges[0][0])
    rt_slow /= (ranges[0][1]-ranges[0][0])

    # Catch any that are nan
    rt_fast[rt_fast!=rt_fast] = 0.0
    rt_slow[rt_slow!=rt_slow] = 0.0

    fweights = rt_fast/(rt_fast+rt_slow)
    sweights = rt_slow/(rt_fast+rt_slow)

    print "LEN"
    print len(fweights),len(data[2])
    print min(fweights),max(fweights),fweights[fweights!=fweights]

    '''
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
    '''

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(data[0],bins=nbins[0],range=(ranges[0][0],ranges[0][1]),weights=fweights,label='Weighted by FRT')
    plt.xlim(ranges[0][0],ranges[0][1])
    plt.xlabel('keVee')
    plt.legend()
    plt.subplot(1,2,2)
    h,xpts,ypts,xerr,yerr = lch.hist_err(data[0],bins=nbins[0],range=(ranges[0][0],ranges[0][1]),weights=sweights,label='Weighted by SRT')
    plt.xlim(ranges[0][0],ranges[0][1])
    plt.xlabel('keVee')
    plt.legend()

    # Fit the data
    popt, pcov = curve_fit(func, xpts, ypts, sigma=yerr)
    print "npts: %f" % (sum(ypts))
    print popt
    print pcov

    x = np.linspace(min(xpts),max(xpts),1000)
    y = func(x,popt[0],popt[1],popt[2])
    #y = func(x,popt[0],popt[1],popt[2],popt[3])
    plt.plot(x,y)



    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(data[1],bins=nbins[1],range=(ranges[1][0],ranges[1][1]),weights=fweights,label='Weighted by FRT')
    plt.xlim(ranges[1][0],ranges[1][1])
    plt.xlabel('Day')
    plt.legend()
    plt.subplot(1,2,2)
    lch.hist_err(data[1],bins=nbins[1],range=(ranges[1][0],ranges[1][1]),weights=sweights,label='Weighted by SRT')
    plt.xlim(ranges[1][0],ranges[1][1])
    plt.xlabel('Day')
    plt.legend()


    '''
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    lch.hist_err(fweights,bins=50)
    plt.subplot(1,2,2)
    lch.hist_err(sweights,bins=50)
    '''

    #plt.figure()
    #lch.hist_2D(data[0],fweights,xbins=nbins[0],ybins=100,xrange=(ranges[0][0],ranges[0][1]),yrange=(0,0.0001))

    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
