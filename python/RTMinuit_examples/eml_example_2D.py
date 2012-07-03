import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.stats as stats

from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

#import minuit
import RTMinuit as rtminuit

pi = np.pi

################################################################################
# Main
################################################################################
def main():

    ranges = [[2.0,8.0],[0.0,400.0]]
    nbins = [100,100]
    bin_widths = np.ones(len(ranges))
    for w,n,r in zip(bin_widths,nbins,ranges):
        w = (r[1]-r[0])/n
    print bin_widths

    fig0 = plt.figure(figsize=(10,9),dpi=100)
    ax00 = fig0.add_subplot(2,2,1)
    ax01 = fig0.add_subplot(2,2,2)
    ax02 = fig0.add_subplot(2,2,3)

    fig1 = plt.figure(figsize=(10,9),dpi=100)
    ax10 = fig1.add_subplot(2,2,1)
    ax11 = fig1.add_subplot(2,2,2)
    ax12 = fig1.add_subplot(2,2,3)

    #ax2 = fig0.add_subplot(2,2,4)
    #ax0.set_xlim(lo,hi)
    #ax1 = fig0.add_subplot(2,1,2) 
    #ax1.set_xlim(lo,hi)


    ############################################################################
    # Gen some data
    ############################################################################
    mean = 5.0
    sigma = 0.5
    nsig = 10000
    xsig = np.random.normal(mean,sigma,nsig)
    #index = xsig>ranges[0][0]
    #index *= xsig<ranges[0][1]
    #xsig = xsig[index==True]
    print len(xsig)

    sig_exp_slope = 170.0
    ygexp = stats.expon(loc=0.0,scale=sig_exp_slope)
    ysig = ygexp.rvs(nsig)
    #print ysig



    # Bkg
    # Exp
    nbkg = 5000
    bkg_exp_slope = 3.0
    xbkg_exp = stats.expon(loc=ranges[0][0],scale=bkg_exp_slope)
    xbkg = xbkg_exp.rvs(nbkg)
    #print xbkg

    # Flat
    ybkg = (ranges[1][1]-ranges[1][0])*np.random.random(nbkg) + ranges[1][0]

    data = np.array([None,None])
    data[0] = np.array(xsig)
    data[0] = np.append(data[0],xbkg)
    data[1] = np.array(ysig)
    data[1] = np.append(data[1],ybkg)

    # Cut out points outside of region.
    index = np.ones(len(data[0]),dtype=np.int)
    for d,r in zip(data,ranges):
        index *= ((d>r[0])*(d<r[1]))

    print len(data[0])
    print len(data[1])

    print index[index==0]
    print "index: ",len(index[index==True])
    
    for i in xrange(len(data)):
        data[i] = data[i][index==True]

    print len(data[0])
    print len(data[1])

    print ranges
    hdata  = lch.hist_2D(data[0],data[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax00)
    hdatay = lch.hist_err(data[1],range=ranges[1],bins=nbins[1],axes=ax01)
    hdatax = lch.hist_err(data[0],range=ranges[0],bins=nbins[0],axes=ax02)
    ax02.set_xlim(ranges[0])
    ax01.set_ylim(0.0)

    print data
    print len(data)

    ############################################################################
    # Gen some MC
    ############################################################################
    nmc = 100000
    mc = np.array([None,None])
    for i,r in enumerate(ranges):
        mc[i] = (r[1]-r[0])*np.random.random(nmc) + r[0]

    print mc
    print len(mc[0])
    print len(mc[1])

    hmc  = lch.hist_2D(mc[0],mc[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax10)
    hmcy = lch.hist_err(mc[1],range=ranges[1],bins=nbins[1],axes=ax11)
    hmcx = lch.hist_err(mc[0],range=ranges[0],bins=nbins[0],axes=ax12)
    ax12.set_xlim(ranges[0])
    ax11.set_ylim(0.0)

    #plt.show()
    #exit()

    ############################################################################
    # Get the efficiency function
    ############################################################################
    max_val = 0.86786
    #threshold = 0.345
    threshold = 4.345
    sigmoid_sigma = 0.241

    
    '''
    ############################################################################
    # Run through the acceptance.
    ############################################################################
    
    indices = np.zeros(len(data),dtype=np.int)
    for i,pt in enumerate(data):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    data = data[indices==1]
    
    print "num: data: ",len(data)

    indices = np.zeros(len(mc),dtype=np.int)
    for i,pt in enumerate(mc):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    mc = mc[indices==1]
    
    num_acc_mc = len(mc)
    print "num: mc: ",len(mc)

    ############################################################################
    # Plot the data and mc
    ############################################################################
    lch.hist_err(data,bins=nbins,range=(lo,hi),axes=ax0)
    lch.hist_err(mc,bins=nbins,range=(lo,hi),axes=ax1)
    '''

    ############################################################################
    # Fit
    ############################################################################

    params_dict = {}
    params_dict['flag'] = {'fix':True,'start_val':1}
    params_dict['mean'] = {'fix':False,'start_val':4.0}
    params_dict['sigma'] = {'fix':False,'start_val':1.0}
    params_dict['sig_y_slope'] = {'fix':False,'start_val':2.0}
    params_dict['bkg_x_slope'] = {'fix':False,'start_val':2.0}
    params_dict['num_sig'] = {'fix':False,'start_val':100.0,'range':(10,10000)}
    params_dict['num_bkg'] = {'fix':False,'start_val':200.0,'range':(10,10000)}

    #myparams = ['flag','mean','sigma','sig_y_slope','bkg_x_slope','num_sig']
    #myparams += ['num_flat']

    params_names,kwd = dict2kwd(params_dict)

    print kwd
    print params_names

    #exit()

    #f = Minuit_FCN([data,mc],myparams)
    f = Minuit_FCN([data,mc],params_names)

    '''
    kwd = {}
    kwd['flag']=0
    kwd['fix_flag']=True
    kwd['mean']=4.0
    kwd['sigma']=1.0
    kwd['num_gauss']=1000
    kwd['num_flat']=1000
    kwd['limit_num_gauss']=(10,100000)
    kwd['limit_num_flat']=(10,100000)
    '''

    m = rtminuit.Minuit(f,**kwd)

    print m.free_param
    print m.fix_param

    # For maximum likelihood method.
    m.set_up(0.5)

    exit()

    m.migrad()

    print m.values,m.errors

    values = m.values # Dictionary

    print "xgauss: ",len(xgauss)
    print "xflat: ",len(xflat)
    print "ngauss: ",values['num_gauss']*(num_raw_mc/float(num_acc_mc))
    print "nflat: ",values['num_flat']*(num_raw_mc/float(num_acc_mc))
    print "err_ngauss: ",m.errors['num_gauss']*(num_raw_mc/float(num_acc_mc))
    print "err_nflat: ",m.errors['num_flat']*(num_raw_mc/float(num_acc_mc))
    #data.sort()
    #print data
    #mc.sort()
    #print mc
    plt.show()

    exit()

    ############################################################################

    x = np.linspace(lo,hi,1000)

    ############################################################################

    # Efficiency function
    efficiency = sigmoid(x,threshold,sigmoid_sigma,max_val)
    ax1 = fig0.add_subplot(2,1,2) 
    ax1.plot(x,efficiency,'r--',linewidth=2)
    ax1.set_xlim(lo,hi)
    ax1.set_ylim(0.0,1.0)


    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)
    #means = np.array([1.2977,1.1])
    #sigmas = np.array([0.077,0.077])
    #numbers = np.array([638,50])

    lshells = lshell_peaks(means,sigmas,num_decays_in_dataset)
    print lshells
    ytot = np.zeros(1000)
    print means
    #HG_trigger = 0.94
    HG_trigger = 1.00
    for n,cp in zip(num_decays_in_dataset,lshells):
        tempy = cp.pdf(x)
        y = n*cp.pdf(x)*bin_width*efficiency/HG_trigger
        print n,integrate.simps(tempy,x=x),integrate.simps(y,x=x)
        ytot += y
        ax0.plot(x,y,'r--',linewidth=2)
    ax0.plot(x,ytot,'r',linewidth=3)

    ############################################################################
    # Surface term
    ############################################################################
    surf_expon = stats.expon(scale=1.0)
    yorg = surf_expon.pdf(values['exp_slope']*x)
    #yorg = surf_expon.pdf(6.0*x)
    y,surf_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=values['num_exp'],fmt='y-',axes=ax0,efficiency=efficiency)
    ytot += y

    ############################################################################
    # Flat term
    ############################################################################
    yorg = np.ones(len(x))
    y,flat_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=efficiency)
    ytot += y
    

    ############################################################################
    # WIMP-like term
    ############################################################################
    '''
    wimp_expon = stats.expon(scale=1.0)
    yorg = wimp_expon.pdf(2.3*x)
    y,wimp_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=330.0,fmt='g-',axes=ax0,efficiency=efficiency)
    ytot += y
    '''
    
    ############################################################################
    # Total-like term
    ############################################################################
    ax0.plot(x,ytot,'b',linewidth=3)

    

    #data = [events,deltat_mc]
    #m = minuit.Minuit(pdfs.extended_maximum_likelihood_function_minuit,p=p0)
    #print m.values
    #m.migrad()

    plt.figure()
    lch.hist_err(mc,bins=108,range=(lo,hi))


################################################################################
################################################################################
if __name__=="__main__":
    main()
