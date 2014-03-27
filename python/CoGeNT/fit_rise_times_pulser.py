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

    #tag = 'pulser_onelognormal'
    #tag = 'pulser'
    #tag = 'pulser_zoomed_in'
    tag = 'pulser_simulated_Nicole'

    '''
    if args.help:
        parser.print_help()
        exit(-1)
    '''

    ############################################################################
    # Read in the data
    ############################################################################
    #infile_name = 'data/LE.txt'
    #infile_name = 'data/HE.txt'
    #infile_name = 'data/pulser_data.dat' # FROM JUAN, 8/2/13, manually scanned pulser runs.
    infile_name = 'data/pulser_data_325ns.dat' # SIMULATED DATA FROM NICOLE
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

    '''
    plt.figure()
    plt.plot(energies,rise_times,'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)

    plt.figure()
    plt.plot(tdays,rise_times,'o',markersize=1.5)
    plt.yscale('log')
    plt.ylim(0.1,10)
    '''

    ############################################################################
    # Plot the data
    ############################################################################
    ############################################################################
    # Look at the rise-time information.
    ############################################################################

    # Will use this later when trying to figure out the energy dependence of 
    # the log-normal parameters.
    # define our (line) fitting function
    expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]
    errfunc = lambda p, x, y, err: (y - expfunc(p, x)) / err

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

    #for i in range(32,-1,-1):
    figcount = 0
    for i in range(0,24):
        #j = 32-i
        j = i
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
            name = "Energy: %0.2f-%0.2f (keVee)" % (elo,ehi)
            plt.text(0.20,0.75,name,transform=axrt[j].transAxes)
            print "=======-------- E BIN ----------==========="
            print name

        nevents = len(data_to_fit)
        print "Nevents for this fit: ",nevents
        #starting_params = [-0.6,0.6,0.2*nevents,  0.6,0.55,0.8*nevents]
        # For pulser fits
        #starting_params = [-0.1,0.8,0.2*nevents,  0.6,0.55,0.8*nevents]
        '''
        if i==0:
            starting_params = [-0.6,0.6,0.2*nevents,  0.6,0.55,0.8*nevents]
        '''
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
        params_dict['fast_num'] = {'fix':False,'start_val':nevents,'limits':(0.0,1.5*nevents),'error':0.01}

        #params_dict['slow_logn_mean'] = {'fix':False,'start_val':starting_params[3],'limits':(-2,2),'error':0.01}
        #params_dict['slow_logn_sigma'] = {'fix':False,'start_val':starting_params[4],'limits':(0.05,30),'error':0.01}
        #params_dict['slow_num'] = {'fix':False,'start_val':starting_params[5],'limits':(0.0,1.5*nevents),'error':0.01}

        # For the pulser fits
        #params_dict['slow_logn_mean'] = {'fix':True,'start_val':0.000,'limits':(-0.002,0.002),'error':0.000001}
        #params_dict['slow_logn_sigma'] = {'fix':True,'start_val':1.000,'limits':(0.9005,1.002),'error':0.000001}
        #params_dict['slow_num'] = {'fix':True,'start_val':0.001,'limits':(0.0,0.002),'error':0.000001}

        # float them
        params_dict['slow_logn_mean'] = {'fix':False,'start_val':starting_params[3],'limits':(-2,2),'error':0.01}
        params_dict['slow_logn_sigma'] = {'fix':False,'start_val':starting_params[4],'limits':(0.05,30),'error':0.01}
        params_dict['slow_num'] = {'fix':False,'start_val':starting_params[5],'limits':(0.0,1.5*nevents),'error':0.01}
        #params_dict['slow_num'] = {'fix':True,'start_val':0.0,'limits':(0.0,1.5*nevents),'error':0.01}

        # Above some value, lock this down.
        # Juan's pulser data
        #if elo>=2.2:
        # Nicole's simulated data
        if elo>=2.8:
            params_dict['slow_logn_mean'] = {'fix':True,'start_val':0.0,'limits':(-2,2),'error':0.01}
            params_dict['slow_logn_sigma'] = {'fix':True,'start_val':1.0,'limits':(0.05,30),'error':0.01}
            params_dict['slow_num'] = {'fix':True,'start_val':1,'limits':(0.0,1.5*nevents),'error':0.01}

        #'''
        if i==0:
            None
            # From Nicole's simulation.
            #params_dict['fast_logn_mean'] = {'fix':True,'start_val':-0.10,'limits':(-2,2),'error':0.01}
            # From Juan
            #params_dict['fast_logn_mean'] = {'fix':True,'start_val':-0.60,'limits':(-2,2),'error':0.01}
            #params_dict['slow_logn_sigma'] = {'fix':True,'start_val':0.50,'limits':(0.05,30),'error':0.01}
        #'''

        # Try fixing the slow sigma
        #params_dict['slow_logn_sigma'] = {'fix':True,'start_val':0.52,'limits':(-2,2),'error':0.01}

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
            kwd['print_level'] = 0
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

            ypts  = pdfs.lognormal(xpts,values['fast_logn_mean'],values['fast_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['fast_num'],fmt='r--',linewidth=2,axes=axrt[j])
            tot_ypts += y

            ypts  = pdfs.lognormal(xpts,values['slow_logn_mean'],values['slow_logn_sigma'],ranges[2][0],ranges[2][1])
            y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[2],scale=values['slow_num'],fmt='r:',linewidth=2,axes=axrt[j])
            tot_ypts += y

            axrt[j].plot(xpts,tot_ypts,'r',linewidth=2)
            axrt[j].set_ylabel(r'Events')
            axrt[j].set_xlabel(r'Rise time ($\mu$s)')
            axrt[j].set_xlim(0,5.0)
            '''
            name = "Plots/rt_slice_%d.png" % (figcount)
            if j%6==5:
                plt.savefig(name)
                figcount += 1
            '''

            #'''
            if math.isnan(values['fast_logn_mean']) == False:
                starting_params = [ \
                values['fast_logn_mean'], \
                values['fast_logn_sigma'], \
                values['fast_num'], \
                values['slow_logn_mean'], \
                values['slow_logn_sigma'],
                values['slow_num'] \
                ]
            #'''

            expts.append((ehi+elo)/2.0)

        if j%6==5:
            name = "Plots/rt_slice_%s_%d.png" % (tag,j/6)
            plt.savefig(name)

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

            pars = ['fast_logn_mean','fast_logn_sigma','fast_num',\
                    'slow_logn_mean','slow_logn_sigma','slow_num']

            for i,p in enumerate(pars):
                if fe.has_key(p):
                    ypts[i].append(fp[p])
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
        labels = ['narrow','wide']

        # Use all or some of the points
        # For the pulser data
        #index = np.arange(0,9)
        # For Nicole's simulated data
        index = np.arange(1,16)

        
        #xp = np.linspace(min(expts),max(expts),100)
        # For Juan's pulser stuff
        #xp = np.linspace(min(expts),expts[8],100)
        # For Nicole's simulated stuff
        xp = np.linspace(min(expts),expts[17],100)
        expts = np.array(expts)

        fvals2 = plt.figure(figsize=(13,4),dpi=100)

        yfitpts = []

        for k in range(0,3):
            # Some of the broad rise times are set to 0.
            #index0s = ypts[3+k]!=0
            #index0s = np.ones(len(ypts[3+k])).astype(bool)
            index0s = np.ones(16).astype(bool)

            fvals2.add_subplot(1,3,k+1)

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

            if k==0:
                plt.ylabel(r'$\Delta \mu$')
            elif k==1:
                plt.ylabel(r'$\Delta \sigma$')
            elif k==2:
                plt.ylabel(r'# wide/# narrow')

            plt.xlim(0.5,3.5)
            plt.xlabel('Energy (keVee)')


            ########################################################################
            # Fit to exponentials.
            ########################################################################
            pinit = [1,1,1]
            # For Juan's stuff
            '''
            if k==0:
                pinit = [1.0, 1.0, -1.2]
            elif k==1:
                #pinit = [1.0, -1.0, -0.5]
                pinit = [-3.0,0.0015,-0.4]
            elif k==2:
                pinit = [-2.0, 1.0, 2.0]
            '''
            
            # For Nicole's stuff
            if k==0:
                pinit = [1.0, 1.0, -1.2]
            elif k==1:
                #pinit = [1.0, -1.0, -0.5]
                pinit = [-3.0,0.0015,-0.4]
            elif k==2:
                pinit = [-2.0, 1.0, 2.0]
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

        fvals2.subplots_adjust(left=0.10, right=0.98,bottom=0.15,wspace=0.25,hspace=0.25)
        name = 'Plots/rt_summary_%s_1.png' % (tag)
        plt.savefig(name)

        ########################################################################
        # Try to fit the individual distributions.
        ########################################################################
        yfitpts = []
        for i in range(0,6):
            yfitpts.append(np.zeros(len(xp)))

        fvals = plt.figure(figsize=(13,4),dpi=100)
        for k in range(0,3):
            fvals.add_subplot(1,3,k+1)
            for ik in range(0,2):
                nindex = k+3*ik
                #print "HERERERERE"
                #print ypts[nindex]
                #print ypts[nindex][ypts[nindex]!=0]
                plt.errorbar(expts[ypts[nindex]!=0],ypts[nindex][ypts[nindex]!=0],xerr=0.01,yerr=[yerrlo[nindex][ypts[nindex]!=0],yerrhi[nindex][ypts[nindex]!=0]],\
                        fmt='o',ecolor='k',mec='k',mfc=colors[ik],label=labels[ik])

                #'''
                # Use part of the data
                #index0 = np.arange(0,3)
                #index1 = np.arange(7,len(expts))
                #index = np.append(index0,index1)

                # Use all or some of the points
                #index = np.arange(0,len(expts))
                # Juan's pulser data.
                #index = np.arange(0,8)
                # Nicole's simulated data
                index = np.arange(1,16)
                if ik>0:
                    # Juan's pulser data
                    #index = np.arange(0,7)
                    # Nicole's simulated data
                    index = np.arange(1,15)
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
                
                out = leastsq(errfunc, pinit, args=(expts[index], ypts[nindex][index], (yerrlo[nindex][index]+yerrhi[nindex][index])/2.0), full_output=1)
                z = out[0]
                zcov = out[1]
                print "Data points: %d %d [%f,%f,%f]" % (k,ik,z[0],z[1],z[2])
                #print "Data points: %d %d [%f,%f,%f]" % (k,ik,np.sqrt(zcov[0][0]),np.sqrt(zcov[1][1]),np.sqrt(zcov[2][2]))
                yfitpts[nindex] = expfunc(z,xp)
                #print zcov
                plt.plot(xp,yfitpts[nindex],'-',color=colors[ik])
                #'''

            if k==0:
                plt.ylim(-1.5,1.5)
            elif k==1:
                plt.ylim(0,1.5)
            plt.xlabel('Energy (keVee)')
            if k==0:
                plt.ylabel(r'Lognormal $\mu$')
            elif k==1:
                plt.ylabel(r'Lognormal $\sigma$')
            elif k==2:
                plt.ylabel(r'Number of events')
            plt.legend()

        #fval
        '''
        fvals.add_subplot(2,3,4)
        plt.plot(xp,yfitpts[3]-yfitpts[0],'-',color='m')

        fvals.add_subplot(2,3,5)
        plt.plot(xp,yfitpts[4]-yfitpts[1],'-',color='m')

        fvals.add_subplot(2,3,6)
        plt.plot(xp,yfitpts[5]/yfitpts[2],'-',color='m')
        '''

        fvals.subplots_adjust(left=0.10, right=0.98,bottom=0.15,wspace=0.25,hspace=0.25)
        name = 'Plots/rt_summary_%s_0.png' % (tag)
        plt.savefig(name)

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
