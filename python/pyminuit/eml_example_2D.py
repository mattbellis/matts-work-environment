import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.stats as stats

from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

import minuit

pi = np.pi

np.random.seed(100)

################################################################################
# Main
################################################################################
def main():

    ranges = [[2.0,8.0],[0.0,400.0]]
    nbins = [100,100]
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n
    #print bin_widths

    fig0 = plt.figure(figsize=(14,4),dpi=100)
    ax00 = fig0.add_subplot(1,3,1)
    ax01 = fig0.add_subplot(1,3,2)
    ax02 = fig0.add_subplot(1,3,3)

    fig1 = plt.figure(figsize=(14,4),dpi=100)
    ax10 = fig1.add_subplot(1,3,1)
    ax11 = fig1.add_subplot(1,3,2)
    ax12 = fig1.add_subplot(1,3,3)

    ############################################################################
    # Gen some data
    ############################################################################
    mean = 5.0
    sigma = 0.5
    nsig = 1000
    xsig = np.random.normal(mean,sigma,nsig)
    #index = xsig>ranges[0][0]
    #index *= xsig<ranges[0][1]
    #xsig = xsig[index==True]
    #print len(xsig)

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

    num_org_sig = len(xsig)
    num_org_bkg = len(xbkg)

    print "num_org_sig: ",num_org_sig
    print "num_org_bkg: ",num_org_bkg 

    # Cut out points outside of region.
    index = np.ones(len(data[0]),dtype=np.int)
    for d,r in zip(data,ranges):
        index *= ((d>r[0])*(d<r[1]))

    #print len(data[0])
    #print len(data[1])

    #print index[index==0]
    #print "index: ",len(index[index==True])
    
    for i in xrange(len(data)):
        data[i] = data[i][index==True]


    ############################################################################
    # Gen some MC
    ############################################################################
    nmc = 50000
    mc = np.array([None,None])
    for i,r in enumerate(ranges):
        mc[i] = (r[1]-r[0])*np.random.random(nmc) + r[0]

    raw_mc = np.array([mc[0],mc[1]])
    num_raw_mc = len(mc[0])

    ############################################################################
    # Get the efficiency function
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241

    max_val = 1.000
    threshold = 0.345
    sigmoid_sigma = 0.241
    #threshold = 4.345
    
    ############################################################################
    # Run through the acceptance.
    ############################################################################
    
    indices = np.zeros(len(data[0]),dtype=np.int)
    for i,pt in enumerate(data[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    data[0] = data[0][indices==1]
    data[1] = data[1][indices==1]
    
    #print "num: data: ",len(data)

    indices = np.zeros(len(mc[0]),dtype=np.int)
    for i,pt in enumerate(mc[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    mc[0] = mc[0][indices==1]
    mc[1] = mc[1][indices==1]
    
    num_acc_mc = len(mc[0])
    #print "num: mc: ",num_acc_mc


    num_acc_sig = len(data[0])
    num_acc_bkg = len(data[1])

    #print "num_acc_sig: ",num_acc_sig
    #print "num_acc_bkg: ",num_acc_bkg 

    ############################################################################
    # Plot the data and MC
    ############################################################################
    
    hdata  = lch.hist_2D(data[0],data[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax02)
    hdatax = lch.hist_err(data[0],range=ranges[0],bins=nbins[0],axes=ax00)
    hdatay = lch.hist_err(data[1],range=ranges[1],bins=nbins[1],axes=ax01)
    ax00.set_xlim(ranges[0])
    ax01.set_xlim(ranges[1])
    ax00.set_ylim(0.0)
    ax01.set_ylim(0.0)

    #print data
    #print len(data)

    hmc  = lch.hist_2D(mc[0],mc[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax12)
    hmcx = lch.hist_err(mc[0],range=ranges[0],bins=nbins[0],axes=ax10)
    hmcy = lch.hist_err(mc[1],range=ranges[1],bins=nbins[1],axes=ax11)
    ax10.set_xlim(ranges[0])
    ax11.set_xlim(ranges[1])
    ax10.set_ylim(0.0)
    ax11.set_ylim(0.0)
    ax12.set_xlim(ranges[0])
    ax12.set_ylim(ranges[1])

    #plt.show()
    #exit()

    '''
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
    params_dict['var_x'] = {'fix':True,'start_val':0,'limits':(2.0,8.0)}
    params_dict['var_y'] = {'fix':True,'start_val':0,'limits':(0.0,400.0)}
    params_dict['mean'] = {'fix':False,'start_val':4.0,'limits':(0.0,10.0)}
    params_dict['sigma'] = {'fix':False,'start_val':1.0,'limits':(0.1,10.0)}
    params_dict['exp_sig_y'] = {'fix':False,'start_val':2.0,'limits':(0,1000)}
    params_dict['exp_bkg_x'] = {'fix':False,'start_val':2.0,'limits':(0,1000)}
    params_dict['num_sig'] = {'fix':False,'start_val':100.0,'limits':(10,100000)}
    params_dict['num_bkg'] = {'fix':False,'start_val':200.0,'limits':(10,100000)}

    params_names,kwd = dict2kwd(params_dict)

    #f = Minuit_FCN([data,mc],params_names)
    f = Minuit_FCN([data,mc],params_dict)

    m = minuit.Minuit(f,**kwd)

    # For maximum likelihood method.
    m.up = 0.5

    m.printMode = 1

    m.migrad()

    print "Finished fit!!\n"
    print minuit_output(m)

    print "\n"

    print "nsig: ",len(xsig)
    print "nbkg: ",len(xbkg)
    print "ntotdata: ",len(data[0])

    values = m.values # Dictionary

    #print "xgauss: ",len(xgauss)
    #print "xflat: ",len(xflat)
    #print "ngauss: ",values['num_gauss']*(num_raw_mc/float(num_acc_mc))
    #print "nflat: ",values['num_flat']*(num_raw_mc/float(num_acc_mc))
    #print "err_ngauss: ",m.errors['num_gauss']*(num_raw_mc/float(num_acc_mc))
    #print "err_nflat: ",m.errors['num_flat']*(num_raw_mc/float(num_acc_mc))

    #data.sort()
    #print data
    #mc.sort()
    #print mc

    print "num_org_sig: ",num_org_sig
    print "num_org_bkg: ",num_org_bkg 

    print "num_acc_sig: ",num_acc_sig
    print "num_acc_bkg: ",num_acc_bkg 

    print "num_raw_mc: ",num_raw_mc 
    print "num_acc_mc: ",num_acc_mc 

    print "nsig: ",values['num_sig'],values['num_sig']*(num_raw_mc/float(num_acc_mc))
    print "nbkg: ",values['num_bkg'],values['num_bkg']*(num_raw_mc/float(num_acc_mc))

    print "ndata: ",len(data[0])
    print "data_norm_integral:       ",fitfunc(data,m.args,params_names,params_dict).sum()
    print "data_norm_integral/ndata: ",fitfunc(data,m.args,params_names,params_dict).sum()/len(data[0])
    print "acc_norm_integral:       ",fitfunc(mc,m.args,params_names,params_dict).sum()
    print "acc_norm_integral/n_acc: ",fitfunc(mc,m.args,params_names,params_dict).sum()/len(mc[0])
    print "raw_norm_integral:       ",fitfunc(raw_mc,m.args,params_names,params_dict).sum()
    print "raw_norm_integral/n_raw: ",fitfunc(raw_mc,m.args,params_names,params_dict).sum()/len(raw_mc[0])
    nsig = values['num_sig']
    norgsig = nsig*(num_raw_mc/float(num_acc_mc))*(1.0/num_raw_mc)*(fitfunc(raw_mc,m.args,params_names,params_dict).sum())
    print "norgsig: ",norgsig

    ############################################################################
    # Plot on the x-projection
    ############################################################################
    ytot = np.zeros(1000)
    xpts = np.linspace(ranges[0][0],ranges[0][1],1000)

    # Sig
    gauss = stats.norm(loc=values['mean'],scale=values['sigma'])
    ypts = gauss.pdf(xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[0],scale=values['num_sig'],fmt='y--',axes=ax00)
    ytot += y

    # Bkg
    bkg_exp = stats.expon(loc=0.0,scale=values['exp_bkg_x'])
    ypts = bkg_exp.pdf(xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[0],scale=values['num_bkg'],fmt='g--',axes=ax00)
    ytot += y

    ax00.plot(xpts,ytot,'b',linewidth=3)

    ############################################################################
    # Plot on the y-projection
    ############################################################################
    ytot = np.zeros(1000)
    xpts = np.linspace(ranges[1][0],ranges[1][1],1000)

    # Sig
    sig_exp = stats.expon(loc=0.0,scale=values['exp_sig_y'])
    ypts = sig_exp.pdf(xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[1],scale=values['num_sig'],fmt='y--',axes=ax01)
    ytot += y

    # Bkg
    ypts = np.ones(len(xpts))

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[1],scale=values['num_bkg'],fmt='g--',axes=ax01)
    ytot += y

    ax01.plot(xpts,ytot,'b',linewidth=3)

    ############################################################################

    #plt.show()

    exit()

    ############################################################################


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
