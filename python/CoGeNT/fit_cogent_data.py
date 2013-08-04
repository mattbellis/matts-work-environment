import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime,timedelta

import scipy.integrate as integrate

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
    data = cut_events_outside_range(data,ranges)
    data = cut_events_outside_subrange(data,subranges[1],data_index=1)

    if args.verbose:
        print_data(energies,tdays)

    print "data after  range cuts: ",len(data[0]),len(data[1]),len(data[2])

    nevents = float(len(data[0]))

    ############################################################################
    # Pre-calculate the slow and fast log-normal probabilities.
    ############################################################################
    # Fast
    print "Precalculating the fast and slow rise time probabilities........"
    mu = [-0.384584,-0.473217,0.070561]
    sigma = [0.751184,-0.301781,0.047121]
    rt_fast = rise_time_prob(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])

    # Slow
    mu = [0.897153,-0.304876,0.044522]
    sigma = [0.126040,0.220238,-0.032878]
    rt_slow = rise_time_prob(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])

    data.append(rt_fast)
    data.append(rt_slow)

    print "Finished with the fast and slow rise time probabilities........"

    #exit()


    ############################################################################
    # Plot the data
    ############################################################################
    fig0 = plt.figure(figsize=(12,4),dpi=100)
    ax0 = fig0.add_subplot(1,2,1)
    ax1 = fig0.add_subplot(1,2,2)


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
                        color='red',ecolor='red',markersize=2,barsabove=False,capsize=0)

    #plt.show()
    #exit()

    ############################################################################
    # Tweak the spacing on the figures.
    fig0.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None,top=0.85)

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
        params_dict[name] = {'fix':True,'start_val':val}
    for i,val in enumerate(decay_constants):
        name = "ls_dc%d" % (i)
        params_dict[name] = {'fix':True,'start_val':val}
    
    ############################################################################
    # exp0 is going to be the simple WIMP-like signal that can modulate or not
    # exp1 is the surface events.
    ############################################################################

    nsurface = 575.0 # For full 917 days
    #nsurface = 506.0 # For full 917 days
    tot_live_days = 808.0 # For the full 917 days
    #tot_live_days = 447.0 # For the time before the fire
    partial_live_days = 0.0
    for sr in subranges[1]:
        partial_live_days += (sr[1]-sr[0])
    nsurface *= partial_live_days/tot_live_days

    nsurface = 5000.0 # 3yr data.

    # Exp 1 is the surface term
    params_dict['e_exp1'] = {'fix':False,'start_val':3.36,'limits':(0.0,10.0)}
    params_dict['num_exp1'] = {'fix':False,'start_val':nsurface,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':575.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':1.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':1000.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':506.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':400.0,'limits':(0.0,100000.0)}
    params_dict['num_flat'] = {'fix':False,'start_val':900.0,'limits':(0.0,100000.0)}
    #params_dict['num_flat'] = {'fix':False,'start_val':1700.0,'limits':(0.0,2000.0)}
    params_dict['e_exp_flat'] = {'fix':True,'start_val':0.05,'limits':(0.00001,10.0)}
    params_dict['t_exp_flat'] = {'fix':False,'start_val':0.05,'limits':(0.00001,10.0)}

    #params_dict['num_exp0'] = {'fix':False,'start_val':296.0,'limits':(0.0,10000.0)}
    params_dict['num_exp0'] = {'fix':True,'start_val':1.0,'limits':(0.0,10000.0)}
    #params_dict['num_exp0'] = {'fix':False,'start_val':575.0,'limits':(0.0,10000.0)}

    # Exponential term in energy
    if args.fit==0 or args.fit==1 or args.fit==5:
        #params_dict['e_exp0'] = {'fix':False,'start_val':2.51,'limits':(0.0,10.0)}
        params_dict['e_exp0'] = {'fix':False,'start_val':3.36,'limits':(0.0,10.0)}

    # Use the dark matter SHM, WIMPS
    if args.fit==2 or args.fit==3 or args.fit==4 or args.fit==6: 
        params_dict['num_exp0'] = {'fix':True,'start_val':1.0,'limits':(0.0,10000.0)}
        params_dict['mDM'] = {'fix':False,'start_val':10.00,'limits':(5.0,20.0)}
        params_dict['sigma_n'] = {'fix':False,'start_val':2e-41,'limits':(1e-42,1e-38)}
        if args.sigma_n != None:
            params_dict['sigma_n'] = {'fix':True,'start_val':args.sigma_n,'limits':(1e-42,1e-38)}
        if args.mDM != None:
            params_dict['mDM'] = {'fix':True,'start_val':args.mDM,'limits':(5.0,20.0)}

    # Let the exponential modulate as a cos term
    if args.fit==1:
        params_dict['wmod_freq'] = {'fix':True,'start_val':yearly_mod,'limits':(0.0,10000.0)}
        params_dict['wmod_phase'] = {'fix':False,'start_val':0.00,'limits':(-2*pi,2*pi)}
        params_dict['wmod_amp'] = {'fix':False,'start_val':0.20,'limits':(0.0,1.0)}
        params_dict['wmod_offst'] = {'fix':True,'start_val':1.00,'limits':(0.0,10000.0)}

    params_names,kwd = dict2kwd(params_dict)

    f = Minuit_FCN([data],params_dict)

    kwd['print_level'] = 2
    # For maximum likelihood method.
    kwd['errordef'] = 0.5

    m = minuit.Minuit(f,**kwd)

    # Up the tolerance.
    #m.tol = 1.0

    m.migrad()
    #m.hesse()

    print "Finished fit!!\n"

    values = m.values # Dictionary
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
    ############################################################################
    # Energy projections
    if args.fit==0 or args.fit==1 or args.fit==5:
        ypts = np.exp(-values['e_exp0']*expts)
        y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp0'],fmt='g-',axes=ax0,efficiency=eff)
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
        srypts,plot,srxpts = plot_pdf_from_lambda(func,bin_width=bin_widths[0],scale=num_wimps,fmt='k-',linewidth=3,axes=ax0,subranges=[[ranges[0][0],ranges[0][1]]],efficiency=efficiency)
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
    ############################################################################
    if args.fit!=5 and args.fit!=6:
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
    #ypts = np.ones(len(expts))
    #ypts =  np.exp(-values['e_exp_flat']*expts)
    ypts  = pdfs.exp(expts,values['e_exp_flat'],ranges[0][0],ranges[0][1])
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=0.95*values['num_flat'],fmt='m--',axes=ax0,efficiency=eff)
    ypts  = pdfs.exp(expts,5.0,ranges[0][0],ranges[0][1])
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=0.05*values['num_flat'],fmt='m--',axes=ax0,efficiency=eff)

    #ypts =  0.95*np.exp(-values['e_exp_flat']*expts)
    ypts  = 0.95*pdfs.exp(expts,values['e_exp_flat'],ranges[0][0],ranges[0][1])
    ypts  += 0.05*pdfs.exp(expts,5.0,ranges[0][0],ranges[0][1])
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=eff,linewidth=3,linecolor='m')
    eytot += y

    # Time projections
    '''
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    '''
    func = lambda x: np.exp(-values['t_exp_flat']*x)
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=0.95*values['num_flat'],fmt='m--',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    flat_tpts = [tot + y for tot,y in zip(flat_tpts,sr_typts)]
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=0.05*values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]
    flat_tpts = [tot + y for tot,y in zip(flat_tpts,sr_typts)]
    
    #ax1.plot(expts,flat_tpts,'m-',linewidth=2)
    ############################################################################
    # L-shell
    ############################################################################
    if args.fit!=5 and args.fit!=6:
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


        # Total on y/t over all subranges.
        for x,y,lsh,flat in zip(sr_txpts,tot_sr_typts,lshell_toty,flat_tpts):
            ax1.plot(x,lsh,'r-',linewidth=2)
            ax1.plot(x,flat,'m-',linewidth=2)
            ax1.plot(x,y,'b',linewidth=3)

    ax0.plot(expts,eytot,'b',linewidth=3)
    # Total on y/t over all subranges.
    for x,y in zip(sr_txpts,tot_sr_typts):
        ax1.plot(x,y,'b',linewidth=3)


    ############################################################################

    '''
    # For efficiency
    fig1 = plt.figure(figsize=(12,4),dpi=100)
    ax10 = fig1.add_subplot(1,2,1)
    ax11 = fig1.add_subplot(1,2,2)
    fig1.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    # Efficiency function
    efficiency = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    ax10.plot(expts,efficiency,'r--',linewidth=2)
    ax10.set_xlim(ranges[0][0],ranges[0][1])
    ax10.set_ylim(0.0,1.0)
    '''

    ############################################################################
    ax1_2 = ax1.twiny()
    #ax1_2.set_xlabel("Date")
    #dates = ['01/02/1991','01/03/1991','01/04/1991']
    #x = [datetime.strptime(d,'%m/%d/%Y').date() for d in dates]
    x = []
    ndivs = 2
    for i in range(ndivs):
        days = ranges[1][1]/ndivs
        date = start_date + timedelta(days=i*days)
        x.append(date)
    date = start_date + timedelta(days=(i+1)*days)
    x.append(date)
    y = range(len(x)) # many thanks to Kyss Tao for setting me straight here
    #print x
    #print y
    ax1_2.plot(x,y,alpha=0)
    #y = 100.0*np.ones(len(x)) # many thanks to Kyss Tao for setting me straight here

    ax1_2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax1_2.xaxis.set_major_locator(mdates.DayLocator())
    ticks = ax1_2.get_xticks()
    #print ticks
    newticks = []
    nticks = len(ticks)
    ndivs = 10
    for i in range(ndivs):
        newticks.append(ticks[int(i*(nticks/float(ndivs)))])
    ax1_2.set_xticks(newticks)
    fig0.autofmt_xdate()
    labels = ax1_2.get_xticklabels()
    for l in labels:
        l.set_rotation(330)
        l.set_fontsize(8)
        l.set_horizontalalignment('right')
    ############################################################################

    # Format the axes a bit
    ax0.set_xlim(ranges[0])
    #ax0.set_ylim(0.0,values['num_flat']/10)
    ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
    ax0.set_ylabel("Interactions/0.025 keVee",fontsize=12)

    ax1.set_xlim(ranges[1])
    #ax1.set_ylim(0.0,values['num_flat']/13)
    ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
    label = "Interactions/%4.1f days" % (bin_widths[1])
    ax1.set_ylabel(label,fontsize=12)

    #############################################################################
    # What is the peak_wimp_date?
    #############################################################################
    peak = start_date + timedelta(days=peak_wimp_date)
    print "\nPeak WIMP signal occurs on:\n"
    print peak.strftime("%D")

    print "Likelihood: %f" % (final_lh)

    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
