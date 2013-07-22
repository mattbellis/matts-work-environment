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

pi = np.pi
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

np.random.seed(200)

yearly_mod = 2*pi/365.0

################################################################################
# Rise time fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    pdf = None

    x = data

    xlo = params_dict['var_rt']['limits'][0]
    xhi = params_dict['var_rt']['limits'][1]

    tot_pdf = np.zeros(len(x))

    #print "HERE"
    #print data[data<0]
    #print data[data>5.0]

    ############################################################################
    # Log-norm structures
    ############################################################################
    means = []
    sigmas = []
    nums = []

    num_tot = 0.0
    num_tot += p[parnames.index("fast_num")]
    num_tot += p[parnames.index("slow_num")]

    means.append(p[pn.index('fast_logn_mean')])
    means.append(p[pn.index('slow_logn_mean')])
    sigmas.append(p[pn.index('fast_logn_sigma')])
    sigmas.append(p[pn.index('slow_logn_sigma')])
    nums.append(p[pn.index('fast_num')]/num_tot) 
    nums.append(p[pn.index('slow_num')]/num_tot) 

    print means,sigmas,nums

    for n,m,s in zip(nums,means,sigmas): 
        pdf  = pdfs.lognormal(x,m,s,xlo,xhi)
        print pdf
        print pdf[pdf<0]
        pdf *= n
        tot_pdf += pdf


    return tot_pdf



################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf(data,p,parnames,params_dict):

    #print data[0]
    ndata = len(data[0])

    flag = p[parnames.index('flag')]

    # Constrain this.
    num_tot = 0.0
    num_tot += p[parnames.index("fast_num")]
    num_tot += p[parnames.index("slow_num")]

    tot_pdf = fitfunc(data[0],p,parnames,params_dict)

    likelihood_func = (-np.log(tot_pdf)).sum()

    #print num_tot,ndata

    ret = likelihood_func - fitutils.pois(num_tot,ndata)

    return ret

################################################################################



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

    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

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

        ############################################################################
        # Declare the fit parameters
        ############################################################################
        params_dict = {}
        params_dict['flag'] = {'fix':True,'start_val':args.fit} 
        params_dict['var_rt'] = {'fix':True,'start_val':0,'limits':(ranges[2][0],ranges[2][1])}
        params_dict['fast_logn_mean'] = {'fix':False,'start_val':0.5,'limits':(-2,2),'error':0.1}
        params_dict['fast_logn_sigma'] = {'fix':False,'start_val':0.5,'limits':(0.01,3),'error':0.1}
        params_dict['fast_num'] = {'fix':False,'start_val':0.2*nevents,'limits':(0.0,1.5*nevents),'error':0.1}
        params_dict['slow_logn_mean'] = {'fix':False,'start_val':0.5,'limits':(-2,2),'error':0.1}
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':1.0,'limits':(0.01,3),'error':0.1}
        params_dict['slow_num'] = {'fix':False,'start_val':0.8*nevents,'limits':(0.0,1.5*nevents),'error':0.1}

        #figrt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None,top=0.85)
        #plt.show()
        #exit()

        ############################################################################
        # Fit
        ############################################################################

        if i==0:
            params_names,kwd = fitutils.dict2kwd(params_dict)
        
            print data[2]
            f = fitutils.Minuit_FCN([[data[2]]],params_dict,emlf)

            kwd['errordef'] = 0.5
            kwd['print_level'] = 2
            print kwd

            m = minuit.Minuit(f,**kwd)

            m.print_param()

            # For maximum likelihood method.
            #m.errordef = 0.5

            # Up the tolerance.
            #m.tol = 1.0

            #m.print_level = 2

            m.migrad()
            #m.hesse()

            print "Finished fit!!\n"

            values = m.values # Dictionary

    '''
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
    func = lambda x: np.ones(len(x))
    sr_typts,plot,sr_txpts = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    tot_sr_typts = [tot + y for tot,y in zip(tot_sr_typts,sr_typts)]

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
        for x,y,lsh in zip(sr_txpts,tot_sr_typts,lshell_toty):
            ax1.plot(x,lsh,'r-',linewidth=2)
            ax1.plot(x,y,'b',linewidth=3)

    ax0.plot(expts,eytot,'b',linewidth=3)
    # Total on y/t over all subranges.
    for x,y in zip(sr_txpts,tot_sr_typts):
        ax1.plot(x,y,'b',linewidth=3)
    '''



    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
