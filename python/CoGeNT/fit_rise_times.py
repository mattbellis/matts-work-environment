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
    data = [energies.copy(),tdays.copy(),rise_times.copy()]
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

    plt.figure()
    plt.plot(energies,rise_times,'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)

    plt.figure()
    plt.plot(tdays,rise_times,'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)

    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    starting_params = [1.2,1.5,0.2*nevents,  0.1,0.8,0.8*nevents]

    fit_parameters = []
    fit_errors = []
    nevs = []
    axrt = []

    elo = 0.0
    ehi = 1.0
    eoffset = 0.5

    #ewidth = 1.50
    #estep = 1.00

    ewidth = 0.200
    estep = 0.20

    expts = []

    #for j in range(25,0,-1):
    for i in range(0,16):
        #i = j
        #i = 25-j
        if i%6==0:
            figrt = plt.figure(figsize=(12,6),dpi=100)
        axrt.append(figrt.add_subplot(2,3, i%6 + 1))

        #figrt = plt.figure(figsize=(6,4),dpi=100)
        #axrt.append(figrt.add_subplot(1,1,1))

        data_to_fit = []
        #h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)

        if i>=0:
            elo = i*estep + eoffset
            ehi = elo + ewidth
            index0 = data[0]>=elo
            index1 = data[0]< ehi
            print elo,ehi
            index = index0*index1
            data_to_fit = data[2][index]

        if len(data_to_fit)>0:
            lch.hist_err(data_to_fit,bins=nbins[2],range=ranges[2],axes=axrt[i])
            plt.ylim(0)
            plt.xlim(ranges[2][0],ranges[2][1])
            name = "%0.2f-%0.2f" % (elo,ehi)
            plt.text(0.75,0.75,name,transform=axrt[i].transAxes)

        nevents = len(data_to_fit)
        if i==0:
            starting_params = [-0.6,0.6,0.2*nevents,  0.6,0.55,0.8*nevents]
        '''
        if elo>=1.0 and elo<1.2:    
            starting_params = [0.1,0.2,0.3*nevents,  0.2,3.0,0.7*nevents]
        '''
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

        params_dict['fast_logn_mean'] = {'fix':False,'start_val':starting_params[0],'limits':(-2,2),'error':0.01}
        params_dict['fast_logn_sigma'] = {'fix':False,'start_val':starting_params[1],'limits':(0.05,30),'error':0.01}
        params_dict['slow_logn_mean'] = {'fix':False,'start_val':starting_params[3],'limits':(-2,2),'error':0.01}
        params_dict['fast_num'] = {'fix':False,'start_val':starting_params[2],'limits':(0.0,1.5*nevents),'error':0.01}
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':starting_params[4],'limits':(0.05,30),'error':0.01}
        params_dict['slow_num'] = {'fix':False,'start_val':starting_params[5],'limits':(0.0,1.5*nevents),'error':0.01}

        #'''
        if i==0:
            params_dict['fast_logn_mean'] = {'fix':True,'start_val':-0.60,'limits':(-2,2),'error':0.01}
            params_dict['slow_logn_sigma'] = {'fix':True,'start_val':0.50,'limits':(0.05,30),'error':0.01}
        #'''

        # Try fixing the slow sigma
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':0.55,'limits':(-2,2),'error':0.01}

        #figrt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None,top=0.85)
        #figrt.subplots_adjust(left=0.05, right=0.98)
        #figrt.subplots_adjust(left=0.15, right=0.98,bottom=0.15)
        figrt.subplots_adjust(left=0.07, right=0.98,bottom=0.10)
        #plt.show()
        #exit()

        ############################################################################
        # Fit
        ############################################################################

        if i>=0 and len(data_to_fit)>0:
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
            errors = m.errors # Dictionary
            fit_parameters.append(values)
            fit_errors.append(errors)
            nevs.append(len(data_to_fit))

            xpts = np.linspace(ranges[2][0],ranges[2][1],1000)
            tot_ypts = np.zeros(len(xpts))

            ypts  = pdfs.lognormal(xpts,values['fast_logn_mean'],values['fast_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['fast_num'],fmt='g-',axes=axrt[i])
            tot_ypts += y

            ypts  = pdfs.lognormal(xpts,values['slow_logn_mean'],values['slow_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['slow_num'],fmt='r-',axes=axrt[i])
            tot_ypts += y

            axrt[i].plot(xpts,tot_ypts,'b',linewidth=2)
            axrt[i].set_ylabel(r'Events')
            axrt[i].set_xlabel(r'Rise time ($\mu$s)')
            name = "Plots/rt_slice_%d.png" % (i)
            plt.savefig(name)

            if math.isnan(values['fast_logn_mean']) == False:
                starting_params = [ \
                values['fast_logn_mean'], \
                values['fast_logn_sigma'], \
                values['fast_num'], \
                values['slow_logn_mean'], \
                values['slow_logn_sigma'],
                values['slow_num'] \
                ]

            expts.append((ehi+elo)/2.0)

    print fit_parameters
    print nevs
    
    ypts = [[],[],[],[],[],[]]
    yerr = [[],[],[],[],[],[]]
    npts = []

    if len(expts)>0:
        for i,fp,fe,n in zip(xrange(len(nevs)),fit_parameters,fit_errors,nevs):
            print "----------"
            ypts[0].append(fp['fast_logn_mean'])
            ypts[1].append(fp['fast_logn_sigma'])
            ypts[2].append(fp['fast_num']/n)
            ypts[3].append(fp['slow_logn_mean'])
            ypts[4].append(fp['slow_logn_sigma'])
            ypts[5].append(fp['slow_num']/n)

            yerr[0].append(fe['fast_logn_mean'])
            yerr[1].append(fe['fast_logn_sigma'])
            yerr[2].append(fe['fast_num']/n)
            yerr[3].append(fe['slow_logn_mean'])
            yerr[4].append(fe['slow_logn_sigma'])
            yerr[5].append(fe['slow_num']/n)

            npts.append(n)

        for i in xrange(len(ypts)):
            ypts[i] = np.array(ypts[i])

        print ypts
        fvals = plt.figure(figsize=(13,4),dpi=100)
        fvals.add_subplot(1,3,1)
        plt.errorbar(expts,ypts[0],xerr=0.01,yerr=yerr[0],fmt='o',ecolor='k',mec='k',mfc='r',label='fast')
        plt.errorbar(expts,ypts[3],xerr=0.01,yerr=yerr[3],fmt='o',ecolor='k',mec='k',mfc='b',label='slow')
        plt.ylim(-1.5,1.5)
        plt.xlabel('Energy (keVee)')
        plt.ylabel(r'Lognormal $\mu$')
        plt.legend()

        index0 = np.arange(0,3)
        index1 = np.arange(7,len(expts))
        #index1 = np.arange(10,20)
        index = np.append(index0,index1)
        print index

        xp = np.linspace(min(expts),max(expts))
        expts = np.array(expts)

        z = np.polyfit(expts[index],ypts[0][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')

        z = np.polyfit(expts[index],ypts[3][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')


        fvals.add_subplot(1,3,2)
        plt.errorbar(expts,ypts[1],xerr=0.01,yerr=yerr[1],fmt='o',ecolor='k',mec='k',mfc='r',label='fast')
        plt.errorbar(expts,ypts[4],xerr=0.01,yerr=yerr[4],fmt='o',ecolor='k',mec='k',mfc='b',label='slow')
        plt.ylim(0.0,1.0)
        plt.xlabel('Energy (keVee)')
        plt.ylabel(r'Lognormal $\sigma$')
        plt.legend()

        z = np.polyfit(expts[index],ypts[1][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')

        z = np.polyfit(expts[index],ypts[4][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')

        fvals.add_subplot(1,3,3)
        plt.errorbar(expts,ypts[2],xerr=0.01,yerr=yerr[2],fmt='o',ecolor='k',mec='k',mfc='r',label='fast')
        plt.errorbar(expts,ypts[5],xerr=0.01,yerr=yerr[5],fmt='o',ecolor='k',mec='k',mfc='b',label='slow')
        plt.ylim(0.0,1.4)
        plt.xlabel('Energy (keVee)')
        plt.ylabel(r'% of events in bin')
        plt.legend()

        z = np.polyfit(expts[index],ypts[2][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')

        z = np.polyfit(expts[index],ypts[5][index],2)
        print z
        p = np.poly1d(z)
        plt.plot(xp,p(xp),'-')

        fvals.subplots_adjust(left=0.08, right=0.98,bottom=0.15,wspace=0.25)
        plt.savefig('Plots/rt_summary.png')

        np.savetxt('rt_parameters.txt',[expts,ypts[0],ypts[1],ypts[2],ypts[3],ypts[4],ypts[5],npts])

    if not args.batch:
        plt.show()

    #exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
