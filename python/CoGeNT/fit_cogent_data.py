import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datetime import datetime,timedelta

import scipy.integrate as integrate

from cogent_utilities import *
from cogent_pdfs import *
from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

import minuit

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
    parser.add_argument('--turn-off-eff', dest='turn_off_eff', action='store_true',\
            default=False, help='Turn off the efficiency.')
    parser.add_argument('--contours', dest='contours', action='store_true',\
            default=False, help='Calculate and plot the contours.')

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
    infile = open('data/before_fire_LG.dat')
    tdays,energies = get_cogent_data(infile,first_event=first_event,calibration=0)

    if args.verbose:
        print_data(energies,tdays)

    data = [energies.copy(),tdays.copy()]

    print "data before range cuts: ",len(data[0]),len(data[1])

    ############################################################################
    # Declare the ranges.
    ############################################################################
    ranges = [[0.5,3.2],[1.0,459.0]]
    #dead_days = [[68,74], [102,107],[306,308]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
    nbins = [108,15]
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

    ############################################################################
    # Plot the data
    ############################################################################
    fig0 = plt.figure(figsize=(12,4),dpi=100)
    ax0 = fig0.add_subplot(1,2,1)
    ax1 = fig0.add_subplot(1,2,2)

    ax0.set_xlim(ranges[0])
    #ax0.set_ylim(0.0,92.0)
    ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
    ax0.set_ylabel("Events/0.025 keVee",fontsize=12)

    ax1.set_xlim(ranges[1])
    ax1.set_ylim(0.0,220.0)
    ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
    ax1.set_ylabel("Event/30.6 days",fontsize=12)

    lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)

    # Do an acceptance correction of some t-bins by hand.
    tbwidth = (ranges[1][1]-ranges[1][0])/float(nbins[1])
    acc_corr = np.zeros(len(ypts))
    acc_corr[2] = tbwidth/(tbwidth-7.0)
    acc_corr[3] = tbwidth/(tbwidth-6.0)
    acc_corr[10] = tbwidth/(tbwidth-3.0)
    ax1.errorbar(xpts, acc_corr*ypts,xerr=xpts_err,yerr=acc_corr*ypts_err,fmt='o', \
                        color='red',ecolor='red',markersize=2,barsabove=False,capsize=0)

    ############################################################################

    # For efficiency
    fig1 = plt.figure(figsize=(12,4),dpi=100)
    ax10 = fig1.add_subplot(1,2,1)
    ax11 = fig1.add_subplot(1,2,2)

    # Tweak the spacing on the figures.
    fig0.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)
    fig1.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    ############################################################################
    # Set up the efficiency function.
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241

    efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)
    if args.turn_off_eff:
        efficiency = lambda x: 1.0

    ############################################################################
    # Get the information for the lshell decays.
    ############################################################################
    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(458)
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
        params_dict[name] = {'fix':True,'start_val':val}
    for i,val in enumerate(decay_constants):
        name = "ls_dc%d" % (i)
        params_dict[name] = {'fix':True,'start_val':val}

    # Exponential term in energy
    if args.fit==0 or args.fit==1:
        params_dict['e_exp0'] = {'fix':False,'start_val':2.51,'limits':(0.0,10.0)}

    params_dict['e_exp1'] = {'fix':True,'start_val':3.36,'limits':(0.0,10.0)}
    params_dict['num_flat'] = {'fix':False,'start_val':1159.0,'limits':(0.0,100000.0)}

    params_dict['num_exp0'] = {'fix':False,'start_val':296.0,'limits':(0.0,10000.0)}

    # Use the dark matter SHM
    if args.fit==2:
        params_dict['num_exp0'] = {'fix':True,'start_val':1.0,'limits':(0.0,10000.0)}
        params_dict['mDM'] = {'fix':False,'start_val':7.00,'limits':(5.0,10000.0)}

    params_dict['num_exp1'] = {'fix':True,'start_val':575.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':506.0,'limits':(0.0,100000.0)}

    # Let the exponential modulate as a cos term
    if args.fit==1:
        params_dict['wmod_freq'] = {'fix':True,'start_val':yearly_mod,'limits':(0.0,10000.0)}
        params_dict['wmod_phase'] = {'fix':False,'start_val':0.00,'limits':(-2*pi,2*pi)}
        params_dict['wmod_amp'] = {'fix':False,'start_val':0.20,'limits':(0.0,1.0)}
        params_dict['wmod_offst'] = {'fix':True,'start_val':1.00,'limits':(0.0,10000.0)}

    params_names,kwd = dict2kwd(params_dict)

    f = Minuit_FCN([data],params_dict)

    m = minuit.Minuit(f,**kwd)

    # For maximum likelihood method.
    m.up = 0.5

    # Up the tolerance.
    m.tol = 1000.0

    m.printMode = 1

    m.migrad()
    m.hesse()

    print "Finished fit!!\n"

    values = m.values # Dictionary

    ############################################################################
    # Print out some diagnostic information
    ############################################################################
    if args.contours:
        plt.figure()
        cx,cy = contours(m,'e_exp0','num_exp0',1.0,10)
        plt.plot(cx,cy)
        cx,cy = contours(m,'e_exp0','num_exp0',1.2,10)
        plt.plot(cx,cy)

    if args.verbose:
        print_correlation_matrix(m)
        print_covariance_matrix(m)

    print minuit_output(m)

    print "nentries: ",len(data[0])

    '''
    names = []
    for name in params_names:
        if 'num_' in name or 'ncalc' in name:
            names.append(name)
    '''

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
    for sr in subranges[1]:
        sr_txpts.append(np.linspace(sr[0],sr[1],1000))
        tot_sr_typts.append(np.zeros(1000))

    ############################################################################
    # Exponential
    ############################################################################
    # Energy projections
    if args.fit==0 or args.fit==1:
        ypts = np.exp(-values['e_exp0']*expts)
        y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp0'],fmt='g-',axes=ax0,efficiency=eff)
        eytot += y

    # Time projections
    if args.fit==0:
        func = lambda x: np.ones(len(x))
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp0'],fmt='g-',axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    elif args.fit==1:
        func = lambda x: values['wmod_offst'] + values['wmod_amp']*np.cos(values['wmod_freq']*x+values['wmod_phase'])   
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp0'],fmt='g-',axes=ax1,subranges=subranges[1])
        tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]

    # Plot wimp term
    if args.fit==2:
        func = lambda x: plot_wimp_er(x,AGe,7.0,time_range=[1,459])
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[0],scale=120.0,fmt='k-',axes=ax0,subranges=[[0.5,3.2]])
        ax0.plot(sr_txpts,sr_typts,'b',linewidth=3)
        eytot += sr_typts

        func = lambda x: plot_wimp_day(x,AGe,7.0,e_range=[0.5,3.2])
        sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=120.0,fmt='k-',axes=ax1,subranges=[[1,459]])
        ax1.plot(sr_txpts,sr_typts,'b',linewidth=3)

    ############################################################################
    # Second exponential
    ############################################################################
    # Energy projections
    ypts = np.exp(-values['e_exp1']*expts)
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp1'],fmt='y-',axes=ax0,efficiency=eff)
    eytot += y

    # Time projections
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp1'],fmt='y-',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]

    ############################################################################
    # Flat
    ############################################################################
    # Energy projections
    ypts = np.ones(len(expts))
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=eff)
    eytot += y

    # Time projections
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]

    ############################################################################
    # L-shell
    ############################################################################
    # Returns pdfs
    lshell_totx = np.zeros(1000)
    lshell_toty = np.zeros(1000)
    for m,s,n,dc in zip(means,sigmas,num_decays_in_dataset,decay_constants):
        gauss = stats.norm(loc=m,scale=s)
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


    ax0.plot(expts,lshell_totx,'r-',linewidth=2)

    ax0.plot(expts,eytot,'b',linewidth=3)

    # Total on y/t over all subranges.
    for x,y,lsh in zip(sr_txpts,tot_sr_typts,lshell_toty):
        ax1.plot(x,lsh,'r-',linewidth=2)
        ax1.plot(x,y,'b',linewidth=3)



    ############################################################################

    # Efficiency function
    efficiency = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    ax10.plot(expts,efficiency,'r--',linewidth=2)
    ax10.set_xlim(ranges[0][0],ranges[0][1])
    ax10.set_ylim(0.0,1.0)

    plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
