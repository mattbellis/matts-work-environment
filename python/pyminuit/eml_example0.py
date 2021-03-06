import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.integrate as integrate

from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

import minuit

pi = np.pi

np.random.seed(100)

################################################################################
# Read in the CoGeNT data
################################################################################
def main():

    lo = 2.0
    hi = 8.0
    nbins = 100
    bin_width = (hi-lo)/nbins
    print bin_width

    fig0 = plt.figure(figsize=(10,9),dpi=100)
    ax0 = fig0.add_subplot(2,1,1)
    ax0.set_xlim(lo,hi)
    ax1 = fig0.add_subplot(2,1,2) 
    ax1.set_xlim(lo,hi)


    ############################################################################
    # Gen some data
    ############################################################################
    mean = 5.0
    sigma = 0.5
    xgauss = np.random.normal(mean,sigma,1000)
    index = xgauss>lo
    index *= xgauss<hi
    xgauss = xgauss[index==True]
    print len(xgauss)

    xflat = (hi-lo)*np.random.random(5000) + lo

    data = np.array(xgauss)
    data = np.append(data,xflat)

    print data
    print len(data)
    #exit()

    ############################################################################
    # Gen some MC
    ############################################################################
    mc = (hi-lo)*np.random.random(10000) + lo
    num_raw_mc = len(mc)

    ############################################################################
    # Get the efficiency function
    ############################################################################
    #max_val = 0.86786
    #threshold = 0.345
    #sigmoid_sigma = 0.241

    max_val = 1.000
    #threshold = 4.345
    threshold = 0.1
    sigmoid_sigma = 0.241
    
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

    ############################################################################
    # Fit
    ############################################################################

    params_dict = {}
    params_dict['flag'] = {'fix':True,'start_val':0}
    params_dict['mean'] = {'fix':False,'start_val':4.0,'limits':(0,10)}
    params_dict['sigma'] = {'fix':False,'start_val':1.0,'limits':(0.1,10)}
    params_dict['num_gauss'] = {'fix':False,'start_val':1000.0,'limits':(1,100000)}
    params_dict['num_flat'] = {'fix':False,'start_val':5000.0,'limits':(1,100000)}

    params_names,kwd = dict2kwd(params_dict)

    #myparams = ('flag','mean','sigma','num_gauss','num_flat')

    #print "here diagnostics."
    #print len(mc)

    #exit()

    f = Minuit_FCN([[data],[mc]],params_names)

    #kwd = {}
    #kwd['flag']=0
    #kwd['fix_flag']=True
    #kwd['mean']=4.0
    #kwd['sigma']=1.0
    #kwd['num_gauss']=1000
    #kwd['num_flat']=1000
    #kwd['limit_num_gauss']=(10,100000)
    #kwd['limit_num_flat']=(10,100000)

    m = minuit.Minuit(f,**kwd)

    #print m.free_param
    #print m.fix_param

    # For maximum likelihood method.
    m.printMode = 1
    m.up = 0.5

    m.migrad()

    print m.values,m.errors

    print minuit_output(m)

    values = m.values # Dictionary

    print "xgauss: ",len(xgauss)
    print "xflat: ",len(xflat)
    print "ngauss: ",values['num_gauss']*(num_raw_mc/float(num_acc_mc))
    #print "nflat: ",values['num_flat']*(num_raw_mc/float(num_acc_mc))
    print "err_ngauss: ",m.errors['num_gauss']*(num_raw_mc/float(num_acc_mc))
    #print "err_nflat: ",m.errors['num_flat']*(num_raw_mc/float(num_acc_mc))
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
