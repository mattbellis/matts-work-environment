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

    #print means,sigmas,nums

    for n,m,s in zip(nums,means,sigmas): 
        pdf  = pdfs.lognormal(x,m,s,xlo,xhi)
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

    plt.figure()
    plt.plot(energies,rise_times,'o',markersize=0.5)


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

    starting_params = [1.0,1.2,0.6*nevents,  0.1,0.8,0.4*nevents]

    fit_parameters = []
    nevs = []
    axrt = []

    for i in range(0,20):
        if i%5==0:
            figrt = plt.figure(figsize=(13,4),dpi=100)
        axrt.append(figrt.add_subplot(1,5, i%5 + 1))
        data_to_fit = []
        #h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)
        if i==0:
            data_to_fit = data[2]
        elif i==1:
            index0 = data[0]>=0.5
            index1 = data[0]<2.0
            index = index0*index1
            data_to_fit = data[2][index]
        elif i==2:
            index0 = data[0]>=2.0
            index1 = data[0]<4.5
            index = index0*index1
            data_to_fit = data[2][index]
        elif i==3:
            fg = plt.figure(figsize=(11,4),dpi=100)
            fg.add_subplot(1,3,1)
            lch.hist_err(data[2],bins=nbins[2],range=ranges[2])
            fg.add_subplot(1,3,2)
            pdf  = pdfs.lognormal(data[2],0.5,0.5, 0,5)
            print "PDF"
            print pdf
            lch.hist_err(pdf,bins=nbins[2])
            xpts = np.linspace(0,5,1000)
            ypts = pdfs.lognormal(xpts,0.01,0.5, 0,5)
            fg.add_subplot(1,3,3)
            plt.plot(xpts,ypts)
            plt.ylim(0,2)
        elif i>=4:
            width = 0.25
            index0 = data[0]>=(i-3)*0.10 + 0.25
            index1 = data[0]< (i-3)*0.10 + 0.50
            print (i-3)*0.10 + 0.25
            print (i-3)*0.10 + 0.50
            index = index0*index1
            data_to_fit = data[2][index]

        if len(data_to_fit)>0:
            lch.hist_err(data_to_fit,bins=nbins[2],range=ranges[2],axes=axrt[i])
            plt.ylim(0)
            plt.xlim(0,5)

        ############################################################################
        # Declare the fit parameters
        ############################################################################
        params_dict = {}
        params_dict['flag'] = {'fix':True,'start_val':args.fit} 
        params_dict['var_rt'] = {'fix':True,'start_val':0,'limits':(ranges[2][0],ranges[2][1])}
        #params_dict['fast_logn_mean'] = {'fix':False,'start_val':0.005,'limits':(-2,2),'error':0.1}
        #params_dict['fast_logn_sigma'] = {'fix':False,'start_val':0.5,'limits':(0.01,5),'error':0.1}
        #params_dict['fast_num'] = {'fix':False,'start_val':0.2*nevents,'limits':(0.0,1.5*nevents),'error':0.1}
        #params_dict['slow_logn_mean'] = {'fix':False,'start_val':0.5,'limits':(-2,2),'error':0.1}
        #params_dict['slow_logn_sigma'] = {'fix':False,'start_val':1.0,'limits':(0.01,5),'error':0.1}
        #params_dict['slow_num'] = {'fix':False,'start_val':0.8*nevents,'limits':(0.0,1.5*nevents),'error':0.1}

        #starting_params = [1.0,1.2,0.6*nevents,  0.1,0.8,0.4*nevents]

        # Worked for 1.0-1.25
        #params_dict['fast_logn_mean'] = {'fix':False,'start_val':1.000,'limits':(-2,2),'error':0.1}
        #params_dict['fast_logn_sigma'] = {'fix':False,'start_val':1.2,'limits':(0.01,5),'error':0.1}
        #params_dict['fast_num'] = {'fix':False,'start_val':0.6*nevents,'limits':(0.0,1.5*nevents),'error':0.1}
        #params_dict['slow_logn_mean'] = {'fix':False,'start_val':0.1,'limits':(-2,2),'error':0.1}
        #params_dict['slow_logn_sigma'] = {'fix':False,'start_val':0.8,'limits':(0.01,5),'error':0.1}
        #params_dict['slow_num'] = {'fix':False,'start_val':0.4*nevents,'limits':(0.0,1.5*nevents),'error':0.1}

        params_dict['fast_logn_mean'] = {'fix':False,'start_val':starting_params[0],'limits':(-2,2),'error':0.1}
        params_dict['fast_logn_sigma'] = {'fix':False,'start_val':starting_params[1],'limits':(0.01,5),'error':0.1}
        params_dict['fast_num'] = {'fix':False,'start_val':starting_params[2],'limits':(0.0,1.5*nevents),'error':0.1}
        params_dict['slow_logn_mean'] = {'fix':False,'start_val':starting_params[3],'limits':(-2,2),'error':0.1}
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':starting_params[4],'limits':(0.01,5),'error':0.1}
        params_dict['slow_num'] = {'fix':False,'start_val':starting_params[5],'limits':(0.0,1.5*nevents),'error':0.1}

        #figrt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None,top=0.85)
        figrt.subplots_adjust(left=0.01, right=0.98)
        #plt.show()
        #exit()

        ############################################################################
        # Fit
        ############################################################################

        #if i<20 and len(data_to_fit)>0:
        #if i>=4 and i<=6 and len(data_to_fit)>0:
        if i>=4 and i<=20 and len(data_to_fit)>0:
            params_names,kwd = fitutils.dict2kwd(params_dict)
        
            #print data_to_fit
            f = fitutils.Minuit_FCN([[data_to_fit]],params_dict,emlf)

            kwd['errordef'] = 0.5
            kwd['print_level'] = 2
            #print kwd

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
            fit_parameters.append(values)
            nevs.append(len(data_to_fit))

            xpts = np.linspace(0,5,1000)
            tot_ypts = np.zeros(len(xpts))

            ypts  = pdfs.lognormal(xpts,values['fast_logn_mean'],values['fast_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['fast_num'],fmt='g-',axes=axrt[i])
            tot_ypts += y

            ypts  = pdfs.lognormal(xpts,values['slow_logn_mean'],values['slow_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['slow_num'],fmt='r-',axes=axrt[i])
            tot_ypts += y

            axrt[i].plot(xpts,tot_ypts,'b',linewidth=2)

            starting_params = [ \
            values['fast_logn_mean'], \
            values['fast_logn_sigma'], \
            values['fast_num'], \
            values['slow_logn_mean'], \
            values['slow_logn_sigma'],
            values['slow_num'] \
            ]

    print fit_parameters
    print nevs
    
    xpts = []
    ypts = [[],[],[],[],[],[]]

    for i,fp,n in zip(xrange(len(nevs)),fit_parameters,nevs):
        print "----------"
        xpts.append(i*0.10 + 0.5-(0.25/2.0))
        ypts[0].append(fp['fast_logn_mean'])
        ypts[1].append(fp['fast_logn_sigma'])
        ypts[2].append(fp['fast_num']/n)
        ypts[3].append(fp['slow_logn_mean'])
        ypts[4].append(fp['slow_logn_sigma'])
        ypts[5].append(fp['slow_num']/n)

    print ypts
    fvals = plt.figure(figsize=(11,4),dpi=100)
    fvals.add_subplot(1,3,1)
    plt.errorbar(xpts,ypts[0],xerr=0.01,yerr=0.01,fmt='o',mfc='r')
    plt.errorbar(xpts,ypts[3],xerr=0.01,yerr=0.01,fmt='o',mfc='b')
    fvals.add_subplot(1,3,2)
    plt.errorbar(xpts,ypts[1],xerr=0.01,yerr=0.01,fmt='o',mfc='r')
    plt.errorbar(xpts,ypts[4],xerr=0.01,yerr=0.01,fmt='o',mfc='b')
    fvals.add_subplot(1,3,3)
    plt.errorbar(xpts,ypts[2],xerr=0.01,yerr=0.01,fmt='o',mfc='r')
    plt.errorbar(xpts,ypts[5],xerr=0.01,yerr=0.01,fmt='o',mfc='b')

    if not args.batch:
        plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
