import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pprint

from datetime import datetime,timedelta

import scipy.integrate as integrate
from scipy import interpolate


import parameters 
from cogent_utilities import *
from cogent_pdfs import *
from fitting_utilities import *
from lichen.plotting_utilities import *
from plotting_utilities import plot_wimp_er
from plotting_utilities import plot_wimp_day

import lichen.lichen as lch

import iminuit as minuit

import argparse

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
    parser.add_argument('--mDM', dest='mDM', type=float,\
            default=None, help='Value of mDM (Mass of DM particle).')
    parser.add_argument('--spike', dest='spikemass', type=float,\
            default=None, help='Mass of a spike allowed to modulate.')
    parser.add_argument('--tag', dest='tag', type=str,\
            default='bkg_only', help='Tag to append to output files and figures.')
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

    #############################################################################
    # Make sure plots don't pop up!
    #############################################################################
    if args.batch:
        plt.switch_backend('Agg')

    ############################################################################
    # Declare the ranges.
    ############################################################################
    ranges,subranges,nbins = parameters.fitting_parameters(args.fit)
    
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n

    ############################################################################
    # Read in the data
    ############################################################################
    #infile = open('data/before_fire_LG.dat')

    #infile_name = 'data/low_gain.txt'
    #infile_name = 'data/high_gain.txt'
    #tdays,energies = get_cogent_data(infile_name,first_event=first_event,calibration=0)

    infile_name = 'data/LE.txt'
    tdays,energies,rise_time = get_3yr_cogent_data(infile_name,first_event=first_event,calibration=0)
    #print tdays
    #print energies
    #print rise_time
    #exit()


    if args.fit==5 or args.fit==6:
        infile_name = 'data/cogent_mc.dat'
        tdays,energies = get_cogent_data(infile_name,first_event=first_event,calibration=999)

    print energies
    if args.verbose:
        print_data(energies,tdays)

    ############################################################################
    # Bundle up the data into one list.
    ############################################################################
    #data = [energies.copy(),tdays.copy()]
    #print "data before range cuts: ",len(data[0]),len(data[1])

    # 3yr data
    #data = [energies.copy(),tdays.copy(),rise_time.copy()]
    data = [energies.copy(),tdays.copy(),rise_time.copy()]
    print "data before range cuts: ",len(data[0]),len(data[1]),len(data[2])
    #exit()

    ############################################################################
    # Cut events out that fall outside the range.
    ############################################################################
    print min(data[1]),max(data[1])
    print min(data[2]),max(data[2])
    print data[2][data[2]<0]

    print len(data[0])
    data = cut_events_outside_range(data,ranges)
    print len(data[0])
    data = cut_events_outside_subrange(data,subranges[1],data_index=1)
    print len(data[0])

    print min(data[1]),max(data[1])
    print min(data[2]),max(data[2])
    print data[2][data[2]<0]
    #exit()

    if args.verbose:
        print_data(energies,tdays)

    print "data after  range cuts: ",len(data[0]),len(data[1]),len(data[2])

    nevents = float(len(data[0]))

    num_wimps = 0.0
    ############################################################################
    # Pre-calculate the slow and fast log-normal probabilities.
    ############################################################################
    # Fast
    print "Precalculating the fast and slow rise time probabilities........"
    #mu = [-0.384584,-0.473217,0.070561]
    #sigma = [0.751184,-0.301781,0.047121]
    # Slow
    #mu = [0.897153,-0.304876,0.044522]
    #sigma = [0.126040,0.220238,-0.032878]
    #rt_slow = rise_time_prob(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])
    ############################################################################
    # From the fit using
    # uncertainty mu = 20%
    # uncertainty sigma = 20%
    # slow sigma = floating
    ############################################################################
    # Parameters for the exponential form for the narrow fast peak.
    #mu0 =  [1.016749,0.786867,-1.203125]
    #sigma0 =  [2.372789,1.140669,0.262251]
    # The entries for the relationship between the broad and narrow peak.
    #fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
    #fast_sigma_rel_k = [-3.667547,0.000256,-0.364826]
    #fast_num_rel_k =  [-2.831665,0.023649,1.144240]

    # Trial 6
    # Using rt 0-8
    #fast_mean_rel_k = [0.792906,-1.628538,-0.201567]
    #fast_sigma_rel_k = [-3.391094,0.000431,-0.369056]
    #fast_num_rel_k = [-3.158560,0.014129,1.229496]

    #mu0 =   [0.701453,0.676855,-1.243412]
    #sigma0 = [2.270888,1.012599,0.272931]

    # Trial 7, 0-8 range, new rels, 0.10 width, removing odd points
    #mu0 =  [0.896497,0.709907,-1.208970]
    #sigma0 = [2.480080,1.215221,0.266656]

    # Using Nicole's simulated stuff
    fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
    fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
    fast_num_rel_k = [-0.261322,5.553102,-5.9144]

    mu0 = [0.374145,0.628990,-1.369876]
    sigma0 = [1.383249,0.495044,0.263360]

    rt_fast = rise_time_prob_fast_exp_dist(data[2],data[0],mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

    # Parameters for the exponential form for the slow peak.
    #mu = [0.945067,0.646431,0.353891]
    #sigma =  [11.765617,94.854276,0.513464]
    # Trial 6
    #mu = [0.846635,0.639263,0.339941]
    #sigma = [0.568532,-0.028607]

    # Trial 7
    #mu = [0.768572,0.588991,0.343744]
    #sigma = [0.566326,-0.031958]

    # Using Nicole's simulated stuff
    mu = [0.269108,0.747275,0.068146]
    sigma = [0.531530,-0.020523]



    rt_slow = rise_time_prob_exp_progression(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])
    ############################################################################

    ####### DO WE NEED THIS FOR THE NORMALIZATION in 2D? #######################
    rt_fast /= (ranges[0][1]-ranges[0][0])
    rt_slow /= (ranges[0][1]-ranges[0][0])

    print (ranges[0][1]-ranges[0][0])

    # FOR DIAGNOSTIC PURPOSES #
    #rt_fast = np.ones(len(rt_fast))
    #rt_slow = np.ones(len(rt_slow))

    #exit()

    print rt_fast
    print rt_slow

    print min(rt_fast),max(rt_fast)
    print min(rt_slow),max(rt_slow)

    print "EHREHRE"
    print rt_fast[rt_fast!=rt_fast]
    print rt_slow[rt_slow!=rt_slow]

    # Catch any that are nan
    rt_fast[rt_fast!=rt_fast] = 0.0
    rt_slow[rt_slow!=rt_slow] = 0.0
    print rt_fast[rt_fast!=rt_fast]
    print rt_slow[rt_slow!=rt_slow]
    #exit()

    data.append(rt_fast)
    data.append(rt_slow)

    ############################################################################
    rt_flat = np.ones(len(rt_slow))/((ranges[2][1]-ranges[2][0])*(ranges[0][1]-ranges[0][0]))
    ############################################################################

    data.append(rt_flat)

    #figrt0 = plt.figure(figsize=(12,4),dpi=100)
    #axrt0 = figrt0.add_subplot(1,1,1)
    #lch.hist_err(data[2],bins=nbins[2],range=ranges[2],axes=axrt0)

    '''
    print "Finished with the fast and slow rise time probabilities........"

    plt.figure()
    plt.plot(data[2],data[3],'o',markersize=1.5)

    plt.figure()
    plt.plot(data[2],data[4],'o',markersize=1.5)

    plt.figure()
    plt.plot(data[0],data[3],'o',markersize=1.5)

    plt.figure()
    plt.plot(data[0],data[4],'o',markersize=1.5)
    '''

    #plt.show()

    #exit()


    ############################################################################
    # Plot the data
    ############################################################################
    fig0a = plt.figure(figsize=(12,6),dpi=100)
    fig0b = plt.figure(figsize=(12,6),dpi=100)
    ax0 = fig0a.add_subplot(1,1,1)
    ax1 = fig0b.add_subplot(1,1,1)

    lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)
    print data[1]
    index0 = data[1]>539.0
    index1 = data[1]<566.4
    index = index0*index1
    print data[1][index]
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)

    # Do an acceptance correction of some t-bins by hand.
    tbwidth = (ranges[1][1]-ranges[1][0])/float(nbins[1])
    print "tbwidth: ",tbwidth
    acc_corr = np.zeros(len(ypts))
    for i,ac in enumerate(acc_corr):
        lo = i*tbwidth + ranges[1][0]
        hi = (i+1)*tbwidth + ranges[1][0]
        #print lo,hi
        vwidth = 0.01
        vals = np.arange(lo,hi,vwidth)
        tot_vals = len(vals)*vwidth
        tot = 0.0
        for sr in subranges[1]:
            #print sr
            for v in vals:
                if v>sr[0] and v<=sr[1]:
                    tot += vwidth
        #print tot,tot_vals
        if tot!=0 and abs(tot-tot_vals)>2.0*vwidth:
            print lo,hi,tot,tot_vals
            #acc_corr[i] = tot_vals/(tot_vals-tot)
            acc_corr[i] = tot_vals/(tot)
            print acc_corr[i]

    print ypts
    print acc_corr
    #exit()

    print acc_corr*ypts
    ax1.errorbar(xpts, acc_corr*ypts,xerr=xpts_err,yerr=acc_corr*ypts_err,fmt='o', \
                        color='red',ecolor='red',markersize=2,barsabove=False,capsize=0,linewidth=2)

    #plt.show()
    #exit()

    ############################################################################
    # Tweak the spacing on the figures.

    #ax1.set_ylim(0.0,420.0)
    #plt.show()
    #exit()
    ############################################################################
    # Set up the efficiency function.
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241

    #eff_scaling = 1.0 # old data
    eff_scaling = 0.9 # 3yr dataset
    efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)/eff_scaling
    if args.turn_off_eff:
        efficiency = lambda x: 1.0

    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    '''
    figrt = plt.figure(figsize=(8,8),dpi=100)
    axrt = []
    for i in range(0,16):
        axrt.append(figrt.add_subplot(4,4,i+1))
        #h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)
        if i==0:
            lch.hist_err(data[2],bins=nbins[2],range=ranges[2],axes=axrt[i])
        elif i==1:
            index0 = data[0]>=0.5
            index1 = data[0]<2.0
            index = index0*index1
            lch.hist_err(data[2][index],bins=nbins[2],range=ranges[2],axes=axrt[i])
        elif i==2:
            index0 = data[0]>=2.0
            index1 = data[0]<4.5
            index = index0*index1
            lch.hist_err(data[2][index],bins=nbins[2],range=ranges[2],axes=axrt[i])
        elif i>=4:
            width = 0.25
            index0 = data[0]>=(i-4)*0.25 + 0.25
            index1 = data[0]<(i-4)*0.25 + 0.50
            index = index0*index1
            lch.hist_err(data[2][index],bins=nbins[2],range=ranges[2],axes=axrt[i])

    #figrt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None,top=0.85)

    #plt.show()
    #exit()

    '''


    ############################################################################
    # Get the information for the lshell decays.
    ############################################################################
    #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data([[1,68],[75,102],[108,306],[309,459]])
    #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data([[1,68],[75,102],[108,306],[309,459],[551,917]])
    #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data([[1,68],[75,102],[108,306],[309,459]])
    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(subranges[1])
    #num_decays_in_dataset *= 0.87 # Might need this to take care of efficiency.

    tot_lshells = num_decays_in_dataset.sum()
    print "Total number of lshells in dataset: ",tot_lshells

    ############################################################################
    # Fit
    ############################################################################

    ############################################################################
    # Declare the fit parameters
    ############################################################################
    params_dict = {}
    params_dict['flag'] = {'fix':True,'start_val':args.fit} 
    params_dict['var_e'] = {'fix':True,'start_val':0,'limits':(ranges[0][0],ranges[0][1])}
    params_dict['var_t'] = {'fix':True,'start_val':0,'limits':(ranges[1][0],ranges[1][1])}

    #######################
    # L-shell parameters
    #######################
    for i,val in enumerate(means):
        name = "ls_mean%d" % (i)
        params_dict[name] = {'fix':True,'start_val':val}
    for i,val in enumerate(sigmas):
        name = "ls_sigma%d" % (i)
        params_dict[name] = {'fix':True,'start_val':val}
    for i,val in enumerate(num_decays_in_dataset):
        name = "ls_ncalc%d" % (i)
        #if i==999:
        if i==2 or i==3:
            params_dict[name] = {'fix':False,'start_val':val,'error':0.01,'limits':(0,50000)}
        else:
            params_dict[name] = {'fix':True,'start_val':val,'error':0.01,'limits':(0,50000)}
    for i,val in enumerate(decay_constants):
        name = "ls_dc%d" % (i)
        params_dict[name] = {'fix':True,'start_val':val,'error':0.01}
    
    ############################################################################
    # exp0 is going to be the simple WIMP-like signal that can modulate or not
    # surf is the surface events.
    ############################################################################

    nsurface = 575.0 # For full 917 days
    #nsurface = 506.0 # For full 917 days
    #tot_live_days = 808.0 # For the full 917 days
    tot_live_days = 1131.0 # For the full 1238 days
    #tot_live_days = 447.0 # For the time before the fire
    partial_live_days = 0.0
    for sr in subranges[1]:
        partial_live_days += (sr[1]-sr[0])
        print partial_live_days

    nsurface *= partial_live_days/tot_live_days

    nsurface = 4400.0 # 3yr data.

    # Exp 1 is the surface term
    #params_dict['e_surf'] = {'fix':False,'start_val':1.0/3.36,'limits':(0.0,10.0)}
    params_dict['k1_surf'] = {'fix':False,'start_val':-0.503,'limits':(-0.7,-0.4),'error':0.1}
    params_dict['k2_surf'] = {'fix':True,'start_val':0.0806,'limits':(0.0,0.2),'error':0.01}
    params_dict['t_surf'] = {'fix':False,'start_val':0.50,'limits':(0.0,10.0),'error':0.01}
    params_dict['num_surf'] = {'fix':False,'start_val':nsurface,'limits':(0.0,100000.0),'error':0.01}

    #params_dict['num_flat'] = {'fix':False,'start_val':3200.0,'limits':(0.0,100000.0),'error':0.01}
    params_dict['num_comp'] = {'fix':False,'start_val':2200.0,'limits':(0.0,100000.0),'error':0.01}
    params_dict['e_exp_flat'] = {'fix':False,'start_val':-0.05,'limits':(0.00001,10.0),'error':0.01}
    params_dict['t_exp_flat'] = {'fix':False,'start_val':0.001,'limits':(0.0000001,10.0),'error':0.01}
    #params_dict['flat_frac'] = {'fix':True,'start_val':0.51,'limits':(0.00001,10.0),'error':0.01}
    #params_dict['flat_frac'] = {'fix':False,'start_val':0.66,'limits':(0.00001,1.0),'error':0.01}

    #params_dict['flat_neutrons_slope'] = {'fix':True,'start_val':0.532,'limits':(0.00001,10.0),'error':0.01}
    #params_dict['flat_neutrons_amp'] = {'fix':True,'start_val':14.0,'limits':(0.00001,10.0),'error':0.01}
    #params_dict['flat_neutrons_offset'] = {'fix':True,'start_val':0.783,'limits':(0.00001,10.0),'error':0.01}
    params_dict['num_neutrons'] = {'fix':False,'start_val':880.0,'limits':(0.0,100000.0),'error':0.01}
    params_dict['flat_neutrons_slope'] = {'fix':True,'start_val':0.920,'limits':(0.00001,10.0),'error':0.01}
    params_dict['flat_neutrons_amp'] = {'fix':True,'start_val':17.4,'limits':(0.00001,10.0),'error':0.01}
    params_dict['flat_neutrons_offset'] = {'fix':True,'start_val':2.38,'limits':(0.00001,10.0),'error':0.01}

    #params_dict['num_exp0'] = {'fix':False,'start_val':296.0,'limits':(0.0,10000.0),'error':0.01}
    params_dict['num_exp0'] = {'fix':True,'start_val':0.0,'limits':(0.0,10000.0),'error':0.01}
    #params_dict['num_exp0'] = {'fix':False,'start_val':575.0,'limits':(0.0,10000.0),'error':0.01}

    # Exponential term in energy
    if args.fit==0 or args.fit==1 or args.fit==5:
        #params_dict['e_exp0'] = {'fix':False,'start_val':2.51,'limits':(0.0,10.0),'error':0.01}
        params_dict['e_exp0'] = {'fix':True,'start_val':0.005,'limits':(0.0,10.0),'error':0.01}

    # Use the dark matter SHM, WIMPS
    if args.fit==2 or args.fit==3 or args.fit==4 or args.fit==6: 
        params_dict['num_exp0'] = {'fix':True,'start_val':0.0,'limits':(0.0,10000.0),'error':0.01}
        params_dict['mDM'] = {'fix':False,'start_val':10.00,'limits':(5.0,40.0),'error':0.01}
        params_dict['sigma_n'] = {'fix':False,'start_val':2e-41,'limits':(1e-42,1e-38),'error':0.01}
        if args.sigma_n != None:
            params_dict['sigma_n'] = {'fix':True,'start_val':args.sigma_n,'limits':(1e-42,1e-38),'error':0.01}
        if args.mDM != None:
            params_dict['mDM'] = {'fix':True,'start_val':args.mDM,'limits':(5.0,40.0),'error':0.01}

    # Let the exponential modulate as a cos term
    if args.fit==1:
        params_dict['wmod_freq'] = {'fix':True,'start_val':yearly_mod,'limits':(0.0,10000.0),'error':0.01}
        params_dict['wmod_phase'] = {'fix':False,'start_val':0.00,'limits':(-2*pi,2*pi),'error':0.01}
        params_dict['wmod_amp'] = {'fix':False,'start_val':0.20,'limits':(0.0,1.0),'error':0.01}
        params_dict['wmod_offst'] = {'fix':True,'start_val':1.00,'limits':(0.0,10000.0),'error':0.01}

    params_dict['num_spike'] = {'fix':True,'start_val':0,'limits':(0.0,500.0),'error':0.01}
    if args.fit == 10:
        params_dict['num_spike'] = {'fix':True,'start_val':200,'limits':(0.0,500.0),'error':0.01}
        params_dict['spike_mass'] = {'fix':True,'start_val':args.spikemass,'limits':(0.0,5.0),'error':0.01}
        params_dict['spike_sigma'] = {'fix':True,'start_val':0.077,'limits':(0.0,1.0),'error':0.01}
        params_dict['spike_freq'] = {'fix':True,'start_val':yearly_mod,'limits':(0.0,10000.0),'error':0.01}
        params_dict['spike_phase'] = {'fix':False,'start_val':0.00,'limits':(-2*pi,2*pi),'error':0.01}
        params_dict['spike_amp'] = {'fix':False,'start_val':0.20,'limits':(0.0,1.0),'error':0.01}
        params_dict['spike_offst'] = {'fix':True,'start_val':1.00,'limits':(0.0,10000.0),'error':0.01}

    params_names,kwd = dict2kwd(params_dict)

    f = Minuit_FCN([data],params_dict)

    kwd['print_level'] = 2
    # For maximum likelihood method.
    kwd['errordef'] = 0.5

    m = minuit.Minuit(f,**kwd)

    values_initial = m.values

    # Up the tolerance.
    #m.tol = 1.0


    m.migrad()
    #m.hesse()

    print "Finished fit!!\n"

    values = m.values # Dictionary
    errors = m.errors # Dictionary
    final_lh = m.fval # 

    ############################################################################
    # Print out some diagnostic information
    ############################################################################
    if args.contours:
        plt.figure()
        cx,cy,cont_values = contours(m,'e_exp0','num_exp0',1.0,10)
        #cx,cy,values = contours(m,'e_exp0','num_exp0',1.0,10)
        print cx
        print cy
        print cont_values
        #plt.plot(cx,cy)
        #cx,cy = contours(m,'e_exp0','num_exp0',1.2,10)
        #plt.plot(cx,cy)

    if args.verbose:
        print_correlation_matrix(m)
        print_covariance_matrix(m)

    #print minuit_output(m)
    #m.print_param()
    #m.print_initial_param()

    print "nentries: ",len(data[0])

    '''
    names = []
    for name in params_names:
        if 'num_' in name or 'ncalc' in name:
            names.append(name)
    '''

    peak_wimp_date = 0
    peak_wimp_val = 0
    ############################################################################
    # Plot the solutions
    ############################################################################
    ############################################################################
    # Plot on the x-projection
    ############################################################################
    expts = np.linspace(ranges[0][0],ranges[0][1],1000)
    txpts = np.linspace(ranges[1][0],ranges[1][1],1000)

    eytot = np.zeros(1000)
    tytot = np.zeros(1000)

    # Get the efficiency over the energy xpts.
    eff = efficiency(expts)

    # Get the points for the subranges
    sr_txpts = []
    tot_sr_typts = []
    flat_tpts = []
    for sr in subranges[1]:
        sr_txpts.append(np.linspace(sr[0],sr[1],1000))
        tot_sr_typts.append(np.zeros(1000))
        flat_tpts.append(np.zeros(1000))

    ############################################################################
    # Exponential
    # This was an early attempt just to use an exponential for a WIMP.
    ############################################################################
    '''
    # Energy projections
    if args.fit==0 or args.fit==1 or args.fit==5:
        ypts = np.exp(-values['e_exp0']*expts)
        y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp0'],fmt='g-',axes=ax0,efficiency=eff) #,label='exponential in energy')
        eytot += y

    # Time projections
    if args.fit==0 or args.fit==5:
        func = lambda x: np.ones(len(x))
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp0'],fmt='g-',axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    elif args.fit==1:
        func = lambda x: values['wmod_offst'] + values['wmod_amp']*np.cos(values['wmod_freq']*x+values['wmod_phase'])   
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp0'],fmt='g-',axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
        # Look for the peak position in the modulation.
        for srty,srtx in zip(sr_typts,sr_txpts):
            if max(srty)>peak_wimp_val:
                peak_wimp_val = max(srty)
                peak_wimp_date = srtx[srty.tolist().index(max(srty))]

    '''

    # Plot wimp term
    if args.fit==2 or args.fit==3 or args.fit==4 or args.fit==6:
        wimp_model = None
        if args.fit==2:
            wimp_model = 'shm'
        elif args.fit==3:
            wimp_model = 'debris'
        elif args.fit==4:
            wimp_model = 'stream'
        elif args.fit==6:
            wimp_model = 'shm'

        num_wimps = 0.0
        for sr in subranges[1]:
            num_wimps += integrate.dblquad(wimp,ranges[0][0],ranges[0][1],lambda x:sr[0],lambda x:sr[1],args=(AGe,values['mDM'],values['sigma_n'],efficiency,wimp_model),epsabs=dblqtol)[0]*(0.333)

        #func = lambda x: plot_wimp_er(x,AGe,values['mDM'],values['sigma_n'],time_range=[1,459],model=wimp_model)
        func = lambda x: plot_wimp_er(x,AGe,values['mDM'],values['sigma_n'],time_range=ranges[1],model=wimp_model)
        srypts,plot,srxpts = plot_pdf_from_lambda(func,bin_width=bin_widths[0],scale=num_wimps,fmt='k-',linewidth=3,axes=ax0,subranges=[[ranges[0][0],ranges[0][1]]],efficiency=efficiency,label='WIMP')
        eytot += srypts[0]

        func = lambda x: plot_wimp_day(x,AGe,values['mDM'],values['sigma_n'],e_range=[ranges[0][0],ranges[0][1]],model=wimp_model)
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=num_wimps,fmt='k-',linewidth=3,axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
        for srty,srtx in zip(sr_typts,sr_txpts):
            if max(srty)>peak_wimp_val:
                peak_wimp_val = max(srty)
                peak_wimp_date = srtx[srty.tolist().index(max(srty))]

        

    ############################################################################
    # Second exponential
    # Surface events
    ############################################################################
    #'''
    if args.fit!=5 and args.fit!=6:
        # Energy projections
        #ypts = np.exp(-values['e_surf']*expts)
        ypts = pdfs.poly(expts, [values['k1_surf'],values['k2_surf']],ranges[0][0],ranges[0][1])
        y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_surf'],fmt='y-',axes=ax0,efficiency=eff,linewidth=4,label='Surface events')
        eytot += y

        # Time projections
        #func = lambda x: np.ones(len(x))
        #sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_surf'],fmt='y-',axes=ax1,subranges=subranges[1])
        func = lambda x: np.exp(-values['t_surf']*x)
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_surf'],fmt='y-',axes=ax1,subranges=subranges[1],linewidth=4)
        surf_ypts = sr_typts
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]

    #'''


    ############################################################################
    # Flat
    ############################################################################
    '''
    # Energy projections
    #ypts = np.ones(len(expts))
    #ypts =  np.exp(-values['e_exp_flat']*expts)
    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['flat_frac']*values['num_flat'],fmt='m--',axes=ax0,efficiency=eff,label='Compton photons from resistor and cosmogenic decays')
    #ypts  = pdfs.exp(expts,0.53,ranges[0][0],ranges[0][1])
    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=(1.0-values['flat_frac'])*values['num_flat'],fmt='c--',axes=ax0,efficiency=eff,label=r'$\mu$-induced $n$ and U,Th,',linewidth=2)


    #ypts = values['flat_frac']*pdfs.exp(expts,values['e_exp_flat'],ranges[0][0],ranges[0][1])
    #ypts += (1.0-values['flat_frac'])*pdfs.exp_plus_flat(expts,values['flat_neutrons_slope'],values['flat_neutrons_amp'],values['flat_neutrons_offset'],ranges[0][0],ranges[0][1])
    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=eff,linewidth=3,linecolor='m',label='Flat total')
    #eytot += y

    # Time projections
    #func = lambda x: np.ones(len(x))
    #sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    #tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    
    func = lambda x: np.exp(-values['t_exp_flat']*x)
    #sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['flat_frac']*values['num_flat'],fmt='m--',axes=ax1,subranges=subranges[1],linewidth=2)
    func = lambda x: np.ones(len(x))
    #sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=(1.0-values['flat_frac'])*values['num_flat'],fmt='c--',axes=ax1,subranges=subranges[1],linewidth=2)

    #ax1.plot(expts,flat_tpts,'m-',linewidth=2)
    '''

    ############################################################################
    # Neutrons from muons and alphas (in the rock)
    ############################################################################
    # Energy
    ypts  = pdfs.exp_plus_flat(expts,values['flat_neutrons_slope'],values['flat_neutrons_amp'],values['flat_neutrons_offset'],ranges[0][0],ranges[0][1])
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_neutrons'],fmt='c-',axes=ax0,efficiency=eff,label=r'($\mu,n$) and ($\alpha,n$) induced neutrons',linewidth=4)
    eytot += y

    # Time
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_neutrons'],fmt='c-',axes=ax1,subranges=subranges[1],linewidth=4)
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    #flat_tpts = [tot + y for tot,y in zip(flat_tpts,sr_typts)]
    
    ############################################################################
    # Comptons
    ############################################################################
    #'''
    # Energy
    ypts  = pdfs.exp(expts,values['e_exp_flat'],ranges[0][0],ranges[0][1])
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_comp'],fmt='m-',axes=ax0,efficiency=eff,label='Comptons: resistor and cosmogenic activations',linewidth=4)
    eytot += y

    # Time
    func = lambda x: np.exp(-values['t_exp_flat']*x)
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_comp'],fmt='m-',axes=ax1,subranges=subranges[1],linewidth=4)
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    #flat_tpts = [tot + y for tot,y in zip(flat_tpts,sr_typts)]
    #'''

    ############################################################################
    # ``spike" term.
    ############################################################################
    if args.fit==10:
        gauss = stats.norm(loc=values['spike_mass'],scale=values['spike_sigma'])
        eypts = gauss.pdf(expts)

        # Energy distributions
        y,plot = plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=values['num_spike'],fmt='k-',axes=ax0,efficiency=eff,linewidth=3,label='spike')
        eytot += y

        # Time distribution
        func = lambda x: values['spike_offst'] + values['spike_amp']*np.cos(values['spike_freq']*x+values['spike_phase'])   
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_spike'],fmt='k-',linewidth=3,axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    ############################################################################
    # L-shell
    ############################################################################
    #'''
    if args.fit!=5 and args.fit!=6:
        # Returns pdfs
        lshell_totx = np.zeros(1000)
        lshell_toty = np.zeros(1000)
        #for m,s,n,dc in zip(means,sigmas,num_decays_in_dataset,decay_constants):
        for i,(mean,s,dc) in enumerate(zip(means,sigmas,decay_constants)):

            name = "ls_ncalc%d" % (i)
            n = values[name]
            gauss = stats.norm(loc=mean,scale=s)
            eypts = gauss.pdf(expts)

            # Energy distributions
            y,plot = plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=n,fmt='r--',axes=ax0,efficiency=eff)
            eytot += y
            lshell_totx += y

            # Time distributions
            func = lambda x: np.exp(dc*x)
            sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=n,fmt='r--',axes=ax1,subranges=subranges[1])
            tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
            lshell_toty = [tot + y for tot,y in zip(lshell_toty,sr_typts)]


        ax0.plot(expts,lshell_totx,'r-',linewidth=4,label='L-shell decays')


        # Total on y/t over all subranges.
        for x,y,lsh,flat in zip(sr_txpts,tot_sr_typts,lshell_toty,flat_tpts):
            ax1.plot(x,lsh,'r-',linewidth=4)
            #ax1.plot(x,flat,'m-',linewidth=2)

    #'''
    ############################################################################
    # Plot the total of all the contributions
    ############################################################################
    #'''
    ax0.plot(expts,eytot,'b',linewidth=4,label='Total')

    # Total on y/t over all subranges.
    for x,y in zip(sr_txpts,tot_sr_typts):
        ax1.plot(x,y,'b',linewidth=4)

    #'''
    '''
    ###########################################################
    # Subtract the fit results to see what's left.
    ###########################################################
    figsub = plt.figure()
    axsub = figsub.add_subplot(1,1,1)
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=axsub)
    tempx = np.array([])
    tempy = np.array([])
    #for x,y,lsh,flat in zip(sr_txpts,tot_sr_typts,lshell_toty,flat_tpts):
    for x,y,lsh,flat in zip(sr_txpts,surf_ypts,lshell_toty,flat_tpts):
        tempx = np.append(tempx,x)
        tempy = np.append(tempy,y+lsh)
    print tempx
    print tempy
    fit_result = interpolate.interp1d(tempx,tempy)
    ypts_sub = ypts - fit_result(xpts)
    acc_corr_sub = acc_corr*ypts - fit_result(xpts)
    ypts_sub_err = ypts_err*ypts_sub/ypts
    axsub.errorbar(xpts,ypts_sub,yerr=ypts_sub_err,fmt='o',
                        color='black',ecolor='black',markersize=2,barsabove=False,capsize=0)
    axsub.errorbar(xpts,acc_corr_sub,yerr=ypts_sub_err,fmt='o',
                        color='red',ecolor='red',markersize=2,barsabove=False,capsize=0)

    #ax1.errorbar(xpts, acc_corr*ypts,xerr=xpts_err,yerr=acc_corr*ypts_err,fmt='o', \
    '''



    ############################################################################

    '''
    # For efficiency
    fig1 = plt.figure(figsize=(12,4),dpi=100)
    ax10 = fig1.add_subplot(1,2,1)
    ax11 = fig1.add_subplot(1,2,2)
    fig1.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    # Efficiency function
    #efficiency = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    #ax10.plot(expts,efficiency,'r--',linewidth=2)
    ax10.plot(expts,eff,'r--',linewidth=2)
    ax10.set_xlim(ranges[0][0],ranges[0][1])
    ax10.set_ylim(0.0,1.1)
    '''

    ############################################################################
    ax1_2 = ax1.twiny()
    #ax1_2.set_xlabel("Date")
    #dates = ['01/02/1991','01/03/1991','01/04/1991']
    #x = [datetime.strptime(d,'%m/%d/%Y').date() for d in dates]
    x = []
    ndivs = 40
    for i in range(ndivs):
        days = ranges[1][1]/ndivs
        #date = start_date + timedelta(days=i*days)
        date = start_date + timedelta(days=i*30)
        x.append(date)
    date = start_date + timedelta(days=(i+1)*days)
    x.append(date)
    y = range(len(x)) # many thanks to Kyss Tao for setting me straight here
    #print x
    #print y
    ax1_2.plot(x,y,alpha=0)
    #y = 100.0*np.ones(len(x)) # many thanks to Kyss Tao for setting me straight here

    ax1_2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax1_2.xaxis.set_major_locator(mdates.MonthLocator())

    #'''
    ticks = ax1_2.get_xticks()
    #print ticks
    newticks = []
    nticks = len(ticks)
    #ndivs = 10
    for i in range(ndivs):
        #newticks.append(ticks[int(i*(nticks/float(ndivs)))])
        if i%3==0:
            newticks.append(ticks[int(i*(nticks/float(ndivs)))])
    ax1_2.set_xticks(newticks)
    #'''
    fig0b.autofmt_xdate()
    labels = ax1_2.get_xticklabels()
    for l in labels:
        l.set_rotation(330)
        l.set_fontsize(12)
        l.set_horizontalalignment('right')
    ############################################################################

    # Format the axes a bit
    ax0.set_xlim(ranges[0])
    #ax0.set_ylim(0.0,values['num_flat']/10)
    #ax0.set_ylim(0.0,220)
    ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
    label = "Interactions/%4.3f keVee" % (bin_widths[0])
    ax0.set_ylabel(label,fontsize=12)
    ax0.legend()
    #ax0.set_yscale('log')
    #ax0.set_ylim(0.1)

    ax1.set_xlim(ranges[1])
    #ax1.set_ylim(0.0,values['num_flat']/8)
    ax1.set_ylim(0.0,350)
    ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
    label = "Interactions/%4.1f days" % (bin_widths[1])
    ax1.set_ylabel(label,fontsize=12)

    fig0a.subplots_adjust(left=0.07, bottom=0.10, right=0.99, wspace=None, hspace=None,top=0.95)
    fig0b.subplots_adjust(left=0.07, bottom=0.12, right=0.99, wspace=None, hspace=None,top=0.85)

    #tag = "background_only"
    #tag = "WIMP_M=10_sigma_n=52e-42"
    tag = args.tag
    name = "Plots/cogent_fit_energy_%s.png" % (tag)
    fig0a.savefig(name)
    name = "Plots/cogent_fit_time_%s.png" % (tag)
    fig0b.savefig(name)
    #############################################################################
    # What is the peak_wimp_date?
    #############################################################################
    peak = start_date + timedelta(days=peak_wimp_date)
    print "\nPeak WIMP signal occurs on:\n"
    print peak.strftime("%D")

    #print "Likelihood: %f\n" % (final_lh)

    #print "num spike: %f" % (values['num_spike'])

    # Nums
    nevents_from_fit = 0
    for v in values:
        if v.find('num')>=0 or v.find('ncalc')>=0:
            nevents_from_fit += values[v]
            print "%-15s %9.4f" % (v,values[v])
    print "%-15s %9.4f" % ('num_wimps',num_wimps)
    nevents_from_fit += num_wimps
    print "\nnevents_from_fit: %f" % (nevents_from_fit)

    print "\n"
    # Not fixed
    for v,iv,e in zip(values,values_initial,errors):
        if not m.is_fixed(v):
            print "%-15s %9.4f +/- %9.4f     %9.4f" % (v,values[v],errors[v],values_initial[v])

    ndata = len(data[0])

    print "\n"
    print "%-15s %15.7f" % ('ndata',ndata)
    print "%-15s %15.7f" % ('ndata fit',nevents_from_fit)
    print "%-15s %15.7f" % ('poisson',pois(nevents_from_fit,ndata))
    print "%-15s %15.7f" % ('max poisson',pois(ndata,ndata))
    print "%-15s %15.7f" % ('lh sans poisson',final_lh+pois(nevents_from_fit,ndata))
    print "%-15s %15.7f" % ('final lh',final_lh)


    name = "fit_results/results_%s.txt" % (tag)
    out_results = open(name,'w')
    #pprint.pprint(values)
    #s = pprint.pformat(values)
    out_results.write(str(values))
    #out_results.write(s)
    out_results.close()

    #print "\nfinal lh: %f" % (final_lh)

    #'''
    if not args.batch:
        plt.show()
    #'''

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
