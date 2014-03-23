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
def funcpoly3(x, a, b, c, d):
    return a + b*x + c*x*x + d*x*x*x

def funcpoly2(x, a, b, c):
    return a + b*x + c*x*x

def funcexp(x, a, b, c):
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
    # YES
    #mu0 =  [1.016749,0.786867,-1.203125]
    #sigma0 =  [2.372789,1.140669,0.262251]

    # Trial 0, ranges = [[0.5,3.2],[1.0,1238.0],[0.0,5.0]]
    # Float sigma for slow
    #mu0 =  [0.916736,0.719082,-1.210973]
    #sigma0 =  [2.456803,1.176434,0.264530]

    # Trial 1, ranges = [[0.5,3.2],[1.0,1238.0],[0.0,5.0]]
    # Fix sigma for slow, 0.55
    #mu0 =  [0.924375,0.799698,-1.216900]
    #sigma0 =  [2.316330,1.120198,0.264719]

    # Trial 2, ranges = [[0.5,3.2],[1.0,1238.0],[0.0,6.0]]
    # Fix sigma for slow, 0.52
    #mu0 =  [1.037549,0.916337,-1.200559]
    #sigma0 =  [2.355926,1.196995,0.268403]

    # Trial 3, using previous fit for starting values
    #mu0 = [0.897636,0.718790,-1.211175]
    #sigma0 = [2.407818,1.143808,0.265886]

    # Trial 4, 0.150 steps
    #mu0 = [0.751306,0.688286,-1.233674]
    #sigma0 =  [2.263108,1.001017,0.271532]

    # Trial 5, 0-8 rt range
    #mu0 =  [0.663503,0.659048,-1.251128]
    #sigma0 = [2.249423,0.971873,0.273167]

    # Trial 6, 0-8 rt range, and new rels
    #mu0 =   [0.701453,0.676855,-1.243412]
    #sigma0 = [2.270888,1.012599,0.272931]

    # Trial 7, 0-8 range, new rels, 0.10 width, removing odd points
    mu0 =  [0.896497,0.709907,-1.208970]
    sigma0 = [2.480080,1.215221,0.266656]

    # The entries for the relationship between the broad and narrow peak.
    # Using rt 0-6
    #fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
    #fast_sigma_rel_k = [-3.667547,0.000256,-0.364826]
    #fast_num_rel_k =  [-2.831665,0.023649,1.144240]

    # Using rt 0-8
    fast_mean_rel_k = [0.792906,-1.628538,-0.201567]
    fast_sigma_rel_k = [-3.391094,0.000431,-0.369056]
    fast_num_rel_k = [-3.158560,0.014129,1.229496]

    rt_fast = rise_time_prob_fast_exp_dist(data[2],data[0],mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

    # Parameters for the exponential form for the slow peak.
    # YES
    #mu = [0.945067,0.646431,0.353891]
    #sigma =  [11.765617,94.854276,0.513464]

    # Trial 0
    #mu =  [0.944007,0.628068,0.362126]
    #sigma = [-0.000349,-29.227369,29.771857]

    # Trial 1
    #mu = [1.137402,0.720385,0.364272]
    #sigma = [0.000000,1.000000,0.52]

    # Trial 2
    #mu =  [1.181151,0.767193,0.372022]
    #sigma = [0.000100,0.001000,0.58]

    # Trial 3
    #mu = [0.938176,0.629022,0.360439]
    ##sigma = [83.292160,-0.016094,0.516246]
    #sigma = [0.547626,-0.017464] # Linear fit.

    # Trial 4
    #mu = [0.867330,0.642910,0.343893]
    ##sigma = [83.292160,-0.016094,0.516246]
    #sigma = [0.548426,-0.017512] # Linear fit.

    # Trial 5
    #mu =  [0.831018,0.634691,0.338035]
    #sigma =  [0.569887,-0.029380]

    # Trial 6
    #mu = [0.846635,0.639263,0.339941]
    #sigma = [0.568532,-0.028607]

    # Trial 7
    mu = [0.768572,0.588991,0.343744]
    sigma = [0.566326,-0.031958]

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
    print sum(fweights),sum(sweights),sum(fweights)+sum(sweights)

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
    popt, pcov = curve_fit(funcpoly2, xpts, ypts, sigma=yerr)
    #popt, pcov = curve_fit(funcexp, xpts, ypts, sigma=yerr,maxfev=10000)
    print "npts: %f" % (sum(ypts))
    print popt
    print pcov

    x = np.linspace(min(xpts),max(xpts),1000)
    y = funcpoly2(x,popt[0],popt[1],popt[2])
    #y = funcpoly3(x,popt[0],popt[1],popt[2],popt[3])
    #y = funcexp(x,popt[0],popt[1],popt[2])
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
