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
    # Declare the ranges.
    ############################################################################
    ranges,subranges,nbins = parameters.fitting_parameters(args.fit)
    
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n

    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    ewidth = 0.150
    estep = 0.150
    estart = 0.5

    x = []
    rt = []
    npts = 300
    nbins = 15
    for i in range(0,nbins):
        emid = i*estep + ewidth/2.0 + estart
        x.append(emid*np.ones(npts))
        rt.append(np.linspace(0.0,8.0,npts))

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
    mu0 =  [0.663503,0.659048,-1.251128]
    sigma0 = [2.249423,0.971873,0.273167]

    # The entries for the relationship between the broad and narrow peak.
    fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
    fast_sigma_rel_k = [-3.667547,0.000256,-0.364826]
    fast_num_rel_k =  [-2.831665,0.023649,1.144240]
    #rt_fast = rise_time_prob_fast_exp_dist(data[2],data[0],mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

    rtfy = []
    for epts,rtpts in zip(x,rt):
        rt_fast = rise_time_prob_fast_exp_dist(rtpts,epts,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])
        rtfy.append(rt_fast)


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
    mu = [0.867330,0.642910,0.343893]
    #sigma = [83.292160,-0.016094,0.516246]
    sigma = [0.548426,-0.017512] # Linear fit.

    # Trial 5
    mu =  [0.831018,0.634691,0.338035]
    sigma =  [0.569887,-0.029380]

    #rt_slow = rise_time_prob_exp_progression(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])
    rtsy = []
    for epts,rtpts in zip(x,rt):
        rt_slow = rise_time_prob_exp_progression(rtpts,epts,mu,sigma,ranges[2][0],ranges[2][1])
        rtsy.append(rt_slow)
    ############################################################################

    axrt = []
    for i in range(0,nbins):
        if i%6==0:
            plt.figure()
        axrt.append(plt.subplot(2,3,i%6+1))
        plt.plot(rt[i],rtfy[i])
        plt.plot(rt[i],rtsy[i])
        name = "%3.2f-%3.2f" % (i*estep+estart,i*estep+estart+ewidth)
        plt.text(0.75,0.75,name,transform=axrt[i].transAxes)


    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
