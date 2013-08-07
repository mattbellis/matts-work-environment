import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime,timedelta

import scipy.integrate as integrate
from scipy.optimize import curve_fit,leastsq

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

# The entries for the narrow peak parameters.
fast_mean0_k = [2.843792,2.265467,-1.096411]
fast_sigma0_k = [3.350738,1.905592,0.224210]
fast_num0_k =  [6.540050,7926.775305,226.999254]

# The entries for the relationship between the broad and narrow peak.
fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
fast_sigma_rel_k = [0.000677,-159.839349,159.382382]
fast_num_rel_k =  [-2.831665,0.023649,1.144240]

#emid = 1.0 # Make this global for ease of fitting.

# Will use this later when trying to figure out the energy dependence of 
# the log-normal parameters.
# define our (line) fitting function
expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]
errfunc = lambda p, x, y, err: (y - expfunc(p, x)) / err

#fast_sigma0_optimal = 1.0
#fast_sigma0_uncert = 1.0

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

    emid = p[pn.index('emid')]

    num_tot = 0.0
    num_tot += p[parnames.index("fast_num")]
    num_tot += p[parnames.index("slow_num")]

    fast_logn_mean0 = p[pn.index('fast_logn_mean0')]
    fast_logn_sigma0 = p[pn.index('fast_logn_sigma0')]
    fast_logn_frac0 = p[pn.index('fast_logn_frac0')]

    slow_logn_mean = p[pn.index('slow_logn_mean')]
    slow_logn_sigma = p[pn.index('slow_logn_sigma')]

    fast_num = p[pn.index('fast_num')]/num_tot
    slow_num = p[pn.index('slow_num')]/num_tot

    #print means,sigmas,nums
    # The entries for the relationship between the broad and narrow peak.
    print "emid: ",emid
    fast_logn_mean_rel = expfunc(fast_mean_rel_k,emid)
    fast_logn_sigma_rel = expfunc(fast_sigma_rel_k,emid)
    fast_logn_num_rel = expfunc(fast_num_rel_k,emid)

    fast_logn_mean1 = fast_logn_mean0 - fast_logn_mean_rel
    fast_logn_sigma1 = fast_logn_sigma0 - fast_logn_sigma_rel
    #fast_num1 = fast_num0 / fast_num_rel

    #fast_logn_frac0 = fast_logn_num0/(fast_num0+fast_num1)
    #print "IN FITFUNC: ",fast_logn_mean0,fast_logn_sigma0,fast_logn_mean1,fast_logn_sigma1

    pdffast0  = pdfs.lognormal(x,fast_logn_mean0,fast_logn_sigma0,xlo,xhi)
    pdffast1  = pdfs.lognormal(x,fast_logn_mean1,fast_logn_sigma1,xlo,xhi)
    pdfslow   = pdfs.lognormal(x,slow_logn_mean, slow_logn_sigma, xlo,xhi)

    tot_pdf = fast_num*(fast_logn_frac0*pdffast0 + (1.0-fast_logn_frac0)*pdffast1) + slow_num*pdfslow
    '''
    for n,m,s in zip(nums,means,sigmas): 
        pdf  = pdfs.lognormal(x,m,s,xlo,xhi)
        pdf *= n
        tot_pdf += pdf
    '''


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

    # GAUSSIAN CONSTRAINT
    mu = p[parnames.index("fast_logn_sigma0")]
    mu0 = p[parnames.index("fast_logn_sigma0_optimal")]
    sig = p[parnames.index("fast_logn_sigma0_uncert")]
    # We are taking the log of the likelihood, so the exponential in the Gaussian function
    # goes away.
    gc = ((mu-mu0)**2)/(2.0*sig*sig)
    #print "Gaussian constraint: ",gc,mu,mu0,sig

    ret -= gc

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
    #infile_name = 'data/pulser_data.dat'
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

    # For the data (two lognormals)
    #starting_params = [-0.6,0.6,0.2*nevents,  0.1,0.8,0.8*nevents]
    # For the pulser fast rise times (two lognormals)
    starting_params = [-0.6,0.5,0.6*nevents,  0.5,0.8,0.4*nevents]


    fit_parameters = []
    fit_errors = []
    fit_mnerrors = []
    nevs = []
    axrt = []

    elo = 0.0
    ehi = 1.0
    eoffset = 0.5

    ewidth = 0.15
    estep = 0.15

    #ewidth = 0.200
    #estep = 0.050

    expts = []

    figcount = 0
    #for i in range(0,16):
    for i in range(16,-1,-1):
        j = 16-i
        #j = i
        if j%6==0:
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
            lch.hist_err(data_to_fit,bins=nbins[2],range=ranges[2],axes=axrt[j])
            plt.ylim(0)
            plt.xlim(ranges[2][0],ranges[2][1])
            name = "%0.2f-%0.2f" % (elo,ehi)
            plt.text(0.75,0.75,name,transform=axrt[j].transAxes)
            print "=======-------- E BIN ----------==========="
            print name

        emid = (elo+ehi)/2.0
        print "HERE ------------------------------- emid: ",emid

        # The entries for the narrow peak parameters.
        fast_mean0 = expfunc(fast_mean0_k,emid)
        fast_sigma0 = expfunc(fast_sigma0_k,emid)
        fast_num0 = expfunc(fast_num0_k,emid)

        # USE THIS FOR THE GAUSSIAN CONSTRAINT
        fast_sigma0_optimal = fast_sigma0
        fast_sigma0_uncert = 0.10*fast_sigma0

        # The entries for the relationship between the broad and narrow peak.
        fast_mean_rel = expfunc(fast_mean_rel_k,emid)
        fast_sigma_rel = expfunc(fast_sigma_rel_k,emid)
        fast_logn_num_rel = expfunc(fast_num_rel_k,emid)

        fast_mean1 = fast_mean0 - fast_mean_rel
        fast_sigma1 = fast_sigma0 - fast_sigma_rel
        fast_num1 = fast_num0 / fast_logn_num_rel

        fast_logn_frac0 = fast_num0/(fast_num0+fast_num1)

        nevents = len(data_to_fit)
        print "Nevents for this fit: ",nevents
        starting_params = [-0.1,0.8,0.2*nevents,  0.6,0.55,0.8*nevents]
        
        ############################################################################
        # Declare the fit parameters
        ############################################################################
        params_dict = {}
        params_dict['flag'] = {'fix':True,'start_val':args.fit} 
        params_dict['var_rt'] = {'fix':True,'start_val':0,'limits':(ranges[2][0],ranges[2][1])}

        params_dict['emid'] = {'fix':True,'start_val':emid,'limits':(ranges[0][0],ranges[0][1])}

        params_dict['fast_logn_mean0'] = {'fix':False,'start_val':fast_mean0,'limits':(-2,2),'error':0.01}
        params_dict['fast_logn_sigma0'] = {'fix':True,'start_val':fast_sigma0,'limits':(0.05,30),'error':0.01}
        params_dict['fast_logn_frac0'] = {'fix':True,'start_val':fast_logn_frac0,'limits':(0.0001,1.0),'error':0.01}

        params_dict['fast_num'] = {'fix':False,'start_val':0.4*nevents,'limits':(0.0,1.5*nevents),'error':0.01}

        params_dict['fast_logn_sigma0_optimal'] = {'fix':True,'start_val':fast_sigma0_optimal}
        params_dict['fast_logn_sigma0_uncert'] = {'fix':True,'start_val':fast_sigma0_uncert}

        #params_dict['fast_logn_mean1'] = {'fix':False,'start_val':starting_params[0],'limits':(-2,2),'error':0.01}
        #params_dict['fast_logn_sigma1'] = {'fix':False,'start_val':starting_params[1],'limits':(0.05,30),'error':0.01}
        #params_dict['fast_num1'] = {'fix':False,'start_val':nevents,'limits':(0.0,1.5*nevents),'error':0.01}

        # float them
        params_dict['slow_logn_mean'] = {'fix':False,'start_val':starting_params[3],'limits':(-2,2),'error':0.01}
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':starting_params[4],'limits':(0.05,30),'error':0.01}
        params_dict['slow_num'] = {'fix':False,'start_val':0.6*nevents,'limits':(0.0,1.5*nevents),'error':0.01}

        # Above some value, lock this down.
        '''
        if elo>=2.2:
            params_dict['slow_logn_mean'] = {'fix':True,'start_val':0.0,'limits':(-2,2),'error':0.01}
            params_dict['slow_logn_sigma'] = {'fix':True,'start_val':1.0,'limits':(0.05,30),'error':0.01}
            params_dict['slow_num'] = {'fix':True,'start_val':1,'limits':(0.0,1.5*nevents),'error':0.01}
        '''

        '''
        if i==0:
            None
            # From Nicole's simulation.
            #params_dict['fast_logn_mean'] = {'fix':True,'start_val':-0.10,'limits':(-2,2),'error':0.01}
            # From Juan
            #params_dict['fast_logn_mean'] = {'fix':True,'start_val':-0.60,'limits':(-2,2),'error':0.01}
            #params_dict['slow_logn_sigma'] = {'fix':True,'start_val':0.50,'limits':(0.05,30),'error':0.01}
        '''

        # Try fixing the slow sigma
        params_dict['slow_logn_sigma'] = {'fix':True,'start_val':0.60,'limits':(-2,2),'error':0.01}

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

            # For maximum likelihood method.
            kwd['errordef'] = 0.5
            kwd['print_level'] = 1
            #print kwd

            m = minuit.Minuit(f,**kwd)

            m.print_param()

            m.migrad()
            #m.hesse()
            m.minos()

            print "Finished fit!!\n"

            values = m.values # Dictionary
            errors = m.errors # Dictionary
            mnerrors = m.get_merrors()
            print "MNERRORS: "
            print mnerrors
            fit_parameters.append(values)
            fit_errors.append(errors)
            fit_mnerrors.append(mnerrors)
            nevs.append(len(data_to_fit))

            xpts = np.linspace(ranges[2][0],ranges[2][1],1000)
            tot_ypts = np.zeros(len(xpts))

            # The entries for the relationship between the broad and narrow peak.
            fast_logn_mean_rel = expfunc(fast_mean_rel_k,values['emid'])
            fast_logn_sigma_rel = expfunc(fast_sigma_rel_k,values['emid'])
            fast_logn_num_rel = expfunc(fast_num_rel_k,values['emid'])

            fast_logn_mean1 = values['fast_logn_mean0'] - fast_logn_mean_rel
            fast_logn_sigma1 = values['fast_logn_sigma0'] - fast_logn_sigma_rel

            tot_ypts_fast = np.zeros(len(xpts))

            ypts  = pdfs.lognormal(xpts,values['fast_logn_mean0'],values['fast_logn_sigma0'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['fast_logn_frac0']*values['fast_num'],fmt='r--',linewidth=2,axes=axrt[j])
            tot_ypts += y
            tot_ypts_fast += y
            ypts  = pdfs.lognormal(xpts,fast_logn_mean1,fast_logn_sigma1,ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=(1.0-values['fast_logn_frac0'])*values['fast_num'],fmt='r--',linewidth=2,axes=axrt[j])
            tot_ypts += y
            tot_ypts_fast += y

            ypts  = pdfs.lognormal(xpts,values['slow_logn_mean'],values['slow_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['slow_num'],fmt='b-',linewidth=2,axes=axrt[j])
            tot_ypts += y

            axrt[j].plot(xpts,tot_ypts_fast,'r-',linewidth=2)

            axrt[j].plot(xpts,tot_ypts,'m',linewidth=2)
            axrt[j].set_ylabel(r'Events')
            axrt[j].set_xlabel(r'Rise time ($\mu$s)')
            name = "Plots/rt_slice_%d.png" % (figcount)
            if j%6==5:
                plt.savefig(name)
                figcount += 1

            #'''
            if math.isnan(values['fast_logn_mean0']) == False:
                starting_params = [ \
                values['fast_logn_mean0'], \
                values['fast_logn_sigma0'], \
                values['fast_num'], \
                values['slow_logn_mean'], \
                values['slow_logn_sigma'],
                values['slow_num'] \
                ]
            #'''

            expts.append((ehi+elo)/2.0)

    #plt.show()
    #exit()
    print fit_parameters
    print nevs
    
    ypts = [[],[],[],[],[],[]]
    yerr = [[],[],[],[],[],[]]
    yerrlo = [[],[],[],[],[],[]]
    yerrhi = [[],[],[],[],[],[]]
    npts = []

    if len(expts)>0:
        #for i,fp,fe,n in zip(xrange(len(nevs)),fit_parameters,fit_errors,nevs):
        for i,fp,fe,n in zip(xrange(len(nevs)),fit_parameters,fit_mnerrors,nevs):
            print "----------"
            #ypts[0].append(fp['fast_logn_mean'])
            #ypts[1].append(fp['fast_logn_sigma'])
            #ypts[2].append(fp['fast_num'])
            #ypts[3].append(fp['slow_logn_mean'])
            #ypts[4].append(fp['slow_logn_sigma'])
            #ypts[5].append(fp['slow_num'])

            pars = ['fast_logn_mean0','fast_logn_sigma0','fast_num',\
                    'slow_logn_mean','slow_logn_sigma','slow_num']

            for i,p in enumerate(pars):
                print "key ",p
                if fe.has_key(p):
                    ypts[i].append(fp[p])
                    print "val: ",fp[p]
                    yerrlo[i].append(abs(fe[p]['lower']))
                    yerrhi[i].append(abs(fe[p]['upper']))
                else:
                    ypts[i].append(0.0)
                    yerrlo[i].append(0.0)
                    yerrhi[i].append(0.0)


            npts.append(n)

        for i in xrange(len(ypts)):
            ypts[i] = np.array(ypts[i])
            yerrlo[i] = np.array(yerrlo[i])
            yerrhi[i] = np.array(yerrhi[i])

        colors = ['r','b']
        labels = ['fast','slow']

        # Use all or some of the points
        index = np.arange(0,9)

        
        #xp = np.linspace(min(expts),max(expts),100)
        xp = np.linspace(min(expts),expts[8],100)
        expts = np.array(expts)

        fvals2 = plt.figure(figsize=(13,6),dpi=100)

        yfitpts = []

        for k in range(0,3):
            # Some of the broad rise times are set to 0.
            #index0s = ypts[3+k]!=0
            #index0s = np.ones(len(ypts[3+k])).astype(bool)
            index0s = np.ones(9).astype(bool)

            fvals2.add_subplot(2,3,k+1)

            tempypts = ypts[0+k]-ypts[3+k]
            # Fractional error
            tempyerrlo = np.sqrt((yerrlo[0+k])**2 + (yerrlo[3+k])**2)
            tempyerrhi = np.sqrt((yerrhi[0+k])**2 + (yerrhi[3+k])**2)
            if k>1:
                tempypts = ypts[0+k][index0s]/ypts[3+k][index0s]
                tempyerrlo = np.sqrt((yerrlo[0+k][index0s]/ypts[3+k][index0s])**2 + (yerrlo[3+k][index0s]*(ypts[0+k][index0s]/(ypts[3+k][index0s]**2)))**2)
                tempyerrhi = np.sqrt((yerrhi[0+k][index0s]/ypts[3+k][index0s])**2 + (yerrhi[3+k][index0s]*(ypts[0+k][index0s]/(ypts[3+k][index0s]**2)))**2)

            plt.errorbar(expts[index0s],tempypts[index0s],xerr=0.01,yerr=[tempyerrlo[index0s],tempyerrhi[index0s]],\
                    fmt='o',ecolor='k',mec='k',mfc='m',label='Ratio')

            plt.xlim(0.5,3.5)


            ########################################################################
            # Fit to exponentials.
            ########################################################################
            pinit = [1,1,1]
            if k==0:
                pinit = [1.0, 1.0, -1.2]
            elif k==1:
                pinit = [1.0, -1.0, -0.5]
            elif k==2:
                pinit = [-2.0, 1.0, 2.0]
            
            print expts[index], tempypts[index], (tempyerrlo[index]+tempyerrhi[index])/2.0
            if sum(tempypts[index]) > 0:
                out = leastsq(errfunc, pinit, args=(expts[index], tempypts[index], (tempyerrlo[index]+tempyerrhi[index])/2.0), full_output=1)
                z = out[0]
                zcov = out[1]
                print "Differences and ratios: %d [%f,%f,%f]" % (k,z[0],z[1],z[2])
                #print "zcov: ",zcov
                if zcov is not None:
                    print "Differences and ratios: %d [%f,%f,%f]" % (k,np.sqrt(zcov[0][0]),np.sqrt(zcov[1][1]),np.sqrt(zcov[2][2]))
                yfitpts = expfunc(z,xp)
                #print zcov
                plt.plot(xp,yfitpts,'-',color='m')



        ########################################################################
        # Try to fit the individual distributions.
        ########################################################################
        yfitpts = []
        for i in range(0,6):
            yfitpts.append(np.zeros(len(xp)))

        fvals = plt.figure(figsize=(13,6),dpi=100)
        for k in range(0,3):
            fvals.add_subplot(2,3,k+1)
            for ik in range(0,2):
                nindex = k+3*ik
                plt.errorbar(expts,ypts[nindex],xerr=0.01,yerr=[yerrlo[nindex],yerrhi[nindex]],\
                        fmt='o',ecolor='k',mec='k',mfc=colors[ik],label=labels[ik])

                #'''
                # Use part of the data
                #index0 = np.arange(0,3)
                #index1 = np.arange(7,len(expts))
                #index = np.append(index0,index1)

                # Use all or some of the points
                #index = np.arange(0,len(expts))
                index = np.arange(0,8)
                if ik>0:
                    index = np.arange(0,7)
                #print index

                ########################################################################
                # Fit to exponentials.
                ########################################################################
                pinit = [1,1,1]
                if ik==0 and k==0:
                    pinit = [1.0, 1.0, -1.2]
                elif ik==0 and k==1:
                    pinit = [4.0, 2.0, 0.0]
                elif ik==0 and k==2:
                    pinit = [2.0, 2000.0, 300.0]
                elif ik==1:
                    pinit = [3.0, 1.5, 0.5]
                
                print "before fit: ",ypts[nindex][index],yerrlo[nindex][index],yerrhi[nindex][index]
                if sum(ypts[nindex][index]) > 0:
                    out = leastsq(errfunc, pinit, args=(expts[index], ypts[nindex][index], (yerrlo[nindex][index]+yerrhi[nindex][index])/2.0), full_output=1)
                    z = out[0]
                    zcov = out[1]
                    print "Data points: %d %d [%f,%f,%f]" % (k,ik,z[0],z[1],z[2])
                    if zcov is not None:
                        print "Data points: %d %d [%f,%f,%f]" % (k,ik,np.sqrt(zcov[0][0]),np.sqrt(zcov[1][1]),np.sqrt(zcov[2][2]))
                    yfitpts[nindex] = expfunc(z,xp)
                    #print zcov
                    plt.plot(xp,yfitpts[nindex],'-',color=colors[ik])

            if k<2:
                plt.ylim(-1.5,1.5)
            plt.xlabel('Energy (keVee)')
            if k==0:
                plt.ylabel(r'Lognormal $\mu$')
            elif k==1:
                plt.ylabel(r'Lognormal $\sigma$')
            elif k==2:
                plt.ylabel(r'Number of events')
            plt.legend()

        #fval
        fvals.add_subplot(2,3,4)
        plt.plot(xp,yfitpts[3]-yfitpts[0],'-',color='m')

        fvals.add_subplot(2,3,5)
        plt.plot(xp,yfitpts[4]-yfitpts[1],'-',color='m')

        fvals.add_subplot(2,3,6)
        plt.plot(xp,yfitpts[5]/yfitpts[2],'-',color='m')

        fvals.subplots_adjust(left=0.08, right=0.98,bottom=0.15,wspace=0.25)
        plt.savefig('Plots/rt_summary.png')

        np.savetxt('rt_parameters.txt',[expts,ypts[0],ypts[1],ypts[2],ypts[3],ypts[4],ypts[5],npts])
        #'''

    #print "Sum ypts[5]: ",sum(ypts[5])

    if not args.batch:
        plt.show()

    #exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
