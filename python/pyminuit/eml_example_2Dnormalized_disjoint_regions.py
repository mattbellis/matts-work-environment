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
    subranges = [[],[[0.0,100.0],[300.0,400.0]]]
    nbins = [100,100]
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n
    #print bin_widths

    fig0 = plt.figure(figsize=(14,4),dpi=100)
    ax00 = fig0.add_subplot(1,3,1)
    ax01 = fig0.add_subplot(1,3,2)
    ax02 = fig0.add_subplot(1,3,3)

    '''
    fig1 = plt.figure(figsize=(14,4),dpi=100)
    ax10 = fig1.add_subplot(1,3,1)
    ax11 = fig1.add_subplot(1,3,2)
    ax12 = fig1.add_subplot(1,3,3)

    fig2 = plt.figure(figsize=(14,4),dpi=100)
    ax20 = fig2.add_subplot(1,3,1)
    ax21 = fig2.add_subplot(1,3,2)
    ax22 = fig2.add_subplot(1,3,3)
    '''

    ############################################################################
    # Gen some data
    ############################################################################
    # Signal
    ############################################################################
    mean = 5.0
    sigma = 0.5
    nsig = 1000
    xsig = np.random.normal(mean,sigma,nsig)

    xsig_copy = np.array(xsig)

    sig_exp_slope = 170.0
    ygexp = stats.expon(loc=0.0,scale=sig_exp_slope)
    ysig = ygexp.rvs(nsig)

    ############################################################################
    # Background
    ############################################################################
    nbkg = 5000
    bkg_exp_slope = 3.0
    xbkg_exp = stats.expon(loc=ranges[0][0],scale=bkg_exp_slope)
    xbkg = xbkg_exp.rvs(nbkg)

    # Flat
    ybkg = (ranges[1][1]-ranges[1][0])*np.random.random(nbkg) + ranges[1][0]

    data = np.array([None,None])
    data[0] = np.array(xsig)
    data[0] = np.append(data[0],xbkg)
    data[1] = np.array(ysig)
    data[1] = np.append(data[1],ybkg)

    num_org_sig = len(xsig)
    num_org_bkg = len(xbkg)

    # Cut out points outside of region.
    index = np.ones(len(data[0]),dtype=np.int)
    for d,r in zip(data,ranges):
        index *= ((d>r[0])*(d<r[1]))


    print "Before in range: ",len(data[0])
    print "Before in range: ",len(data[1])

    for i in xrange(len(data)):
        data[i] = data[i][index==True]

    print "After in range: ",len(data[0])
    print "After in range: ",len(data[1])

    # Cut out points outside of subranges.
    print subranges[1][0][0],subranges[1][0][1]
    index = ((data[1]>subranges[1][0][0])*(data[1]<subranges[1][0][1]))
    index += ((data[1]>subranges[1][1][0])*(data[1]<subranges[1][1][1]))

    for i in xrange(len(data)):
        data[i] = data[i][index==True]

    print "After in range: ",len(data[0])
    print "After in range: ",len(data[1])

    n_org_data = len(data[0])


    ############################################################################
    # Gen some MC
    ############################################################################
    '''
    nmc = 50000
    mc = np.array([None,None])
    for i,r in enumerate(ranges):
        mc[i] = (r[1]-r[0])*np.random.random(nmc) + r[0]

    raw_mc = np.array([mc[0],mc[1]])
    num_raw_mc = len(mc[0])

    nplotmc = 1000000
    plotmc = np.array([None,None])
    for i,r in enumerate(ranges):
        plotmc[i] = (r[1]-r[0])*np.random.random(nplotmc) + r[0]
    '''
    ############################################################################
    # Get the efficiency function
    ############################################################################
    #max_val = 0.86786
    max_val = 1.00
    threshold = 0.345
    sigmoid_sigma = 0.241

    #max_val = 0.86786
    #max_val = 1.000
    #threshold = 0.345
    #sigmoid_sigma = 0.241

    #max_val = 0.86786
    #max_val = 1.0
    #threshold = 4.345
    #threshold = 6.345
    #threshold = 2.345
    #sigmoid_sigma = 0.241
    
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

    '''
    indices = np.zeros(len(mc[0]),dtype=np.int)
    for i,pt in enumerate(mc[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    mc[0] = mc[0][indices==1]
    mc[1] = mc[1][indices==1]
    
    num_acc_mc = len(mc[0])

    #print "num: mc: ",num_acc_mc

    raw_plotmc = np.array([np.array(plotmc[0]),np.array(plotmc[1])])

    indices = np.zeros(len(plotmc[0]),dtype=np.int)
    for i,pt in enumerate(plotmc[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    plotmc[0] = plotmc[0][indices==1]
    plotmc[1] = plotmc[1][indices==1]
    '''
    
    indices = np.zeros(len(xsig_copy),dtype=np.int)
    for i,pt in enumerate(xsig_copy):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    xsig_copy = xsig_copy[indices==1]

    num_surv_signal = len(xsig_copy)


    '''
    # Count how much signal and background made it.
    indices = np.zeros(len(signal[0]),dtype=np.int)
    for i,pt in enumerate(signal[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    print len(signal[0])
    print len(indices)
    signalacc0 = signal[0][indices==1]
    signalacc1 = signal[1][indices==1]
    
    indices = np.zeros(len(background[0]),dtype=np.int)
    for i,pt in enumerate(background[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1
    backgroundacc0 = background[0][indices==1]
    backgroundacc1 = background[1][indices==1]
    

    num_acc_sig = len(signalacc0)
    num_acc_bkg = len(backgroundacc0)

    #print "num_acc_sig: ",num_acc_sig
    #print "num_acc_bkg: ",num_acc_bkg 
    '''

    ############################################################################
    # Plot the data and MC
    ############################################################################
    
    hdata  = lch.hist_2D(data[0],data[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax02)
    hdatax = lch.hist_err(data[0],range=ranges[0],bins=nbins[0],axes=ax00)
    print "hdatax: ",hdatax[2].sum()
    hdatay = lch.hist_err(data[1],range=ranges[1],bins=nbins[1],axes=ax01)
    ax00.set_xlim(ranges[0])
    ax01.set_xlim(ranges[1])
    ax00.set_ylim(0.0)
    ax01.set_ylim(0.0)

    ax02.set_ylim(ranges[1][0],ranges[1][1])
    ax02.set_xlim(ranges[0][0],ranges[0][1])

    #plt.show()

    #print data
    #print len(data)

    '''
    hmc  = lch.hist_2D(mc[0],mc[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax12)
    hmcx = lch.hist_err(mc[0],range=ranges[0],bins=nbins[0],axes=ax10)
    hmcy = lch.hist_err(mc[1],range=ranges[1],bins=nbins[1],axes=ax11)
    ax10.set_xlim(ranges[0])
    ax11.set_xlim(ranges[1])
    ax10.set_ylim(0.0)
    ax11.set_ylim(0.0)
    ax12.set_xlim(ranges[0])
    ax12.set_ylim(ranges[1])
    '''

    '''
    hplotmc  = lch.hist_2D(plotmc[0],plotmc[1],xrange=ranges[0],yrange=ranges[1],xbins=nbins[0],ybins=nbins[1],axes=ax22)
    hplotmcx = lch.hist_err(plotmc[0],range=ranges[0],bins=nbins[0],axes=ax20)
    hplotmcy = lch.hist_err(plotmc[1],range=ranges[1],bins=nbins[1],axes=ax21)
    ax20.set_xlim(ranges[0])
    ax21.set_xlim(ranges[1])
    ax20.set_ylim(0.0)
    ax21.set_ylim(0.0)
    ax22.set_xlim(ranges[0])
    ax22.set_ylim(ranges[1])
    '''

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
    params_dict['flag'] = {'fix':True,'start_val':3}
    params_dict['var_x'] = {'fix':True,'start_val':0,'limits':(2.0,8.0)}
    params_dict['var_y'] = {'fix':True,'start_val':0,'limits':(0.0,400.0)}
    params_dict['mean'] = {'fix':False,'start_val':4.0,'limits':(0.0,10.0)}
    params_dict['sigma'] = {'fix':False,'start_val':0.5,'limits':(0.1,10.0)}
    params_dict['exp_sig_y'] = {'fix':False,'start_val':1.0/200.0,'limits':(0,1000)}
    params_dict['exp_bkg_x'] = {'fix':False,'start_val':2.0,'limits':(0,1000)}
    params_dict['num_sig'] = {'fix':False,'start_val':500.0,'limits':(10,100000)}
    params_dict['num_bkg'] = {'fix':False,'start_val':2000.0,'limits':(10,100000)}

    params_names,kwd = dict2kwd(params_dict)

    #f = Minuit_FCN([data,mc],params_names)
    f = Minuit_FCN([data],params_dict)

    m = minuit.Minuit(f,**kwd)

    # For maximum likelihood method.
    m.up = 0.5

    #print m.maxcalls 
    #m.maxcalls = 10000
    m.strategy = 1

    m.printMode = 1

    m.migrad()

    print "Finished fit!!\n"
    print minuit_output(m)

    print "\n"

    print "nsig: ",len(xsig)
    print "nbkg: ",len(xbkg)
    print "ntotdata: ",len(data[0])

    values = m.values # Dictionary
    
    #exit()

    ndata = len(data[0])
    #test_point = np.array([np.array([2.5,2.5,2.5,2.5,2.5]),np.array([10.0,20,40,100,200])])
    #test_point = np.array([5.0*np.ones(10),np.linspace(0,400,10)])

    '''
    temp_vals = list(m.args)
    temp_vals[params_names.index('num_bkg')] = 0.0
    acc_integral_sig = fitfunc(mc,    temp_vals,params_names,params_dict).sum()
    acc_integral = fitfunc(mc,    temp_vals,params_names,params_dict).sum()
    raw_integral = fitfunc(raw_mc,temp_vals,params_names,params_dict).sum()
    #test_val = fitfunc(test_point,temp_vals,params_names,params_dict)
    #print "test_val: ",test_val,test_val.sum()
    print "integrals: ",acc_integral,raw_integral,raw_integral/acc_integral
    number_of_events = return_numbers_of_events(m,acc_integral,num_acc_mc,raw_integral,num_raw_mc,ndata,['num_sig'])
    print number_of_events
    print "\n"

    temp_vals = list(m.args)
    temp_vals[params_names.index('num_sig')] = 0.0
    acc_integral_bkg = fitfunc(mc,    temp_vals,params_names,params_dict).sum()
    acc_integral = fitfunc(mc,    temp_vals,params_names,params_dict).sum()
    raw_integral = fitfunc(raw_mc,temp_vals,params_names,params_dict).sum()
    #test_val = fitfunc(test_point,temp_vals,params_names,params_dict)
    #print "test_val: ",test_val,test_val.sum()
    print "integrals: ",acc_integral,raw_integral,raw_integral/acc_integral
    number_of_events = return_numbers_of_events(m,acc_integral,num_acc_mc,raw_integral,num_raw_mc,ndata,['num_bkg'])
    print number_of_events
    print "\n"

    acc_integral_tot = fitfunc(mc,    m.args,params_names,params_dict).sum()
    acc_integral = fitfunc(mc,    m.args,params_names,params_dict).sum()
    raw_integral = fitfunc(raw_mc,m.args,params_names,params_dict).sum()
    #test_val = fitfunc(test_point,m.args,params_names,params_dict)
    #print "test_val: ",test_val,test_val.sum()
    print "integrals: ",acc_integral,raw_integral,raw_integral/acc_integral
    number_of_events = return_numbers_of_events(m,acc_integral,num_acc_mc,raw_integral,num_raw_mc,ndata)
    print number_of_events
    print "\n"

    sig_frac = acc_integral_sig/acc_integral_tot
    bkg_frac = acc_integral_bkg/acc_integral_tot
    print "fractions: ",sig_frac,bkg_frac
    '''


    '''
    print "\nm.covariance"
    print m.covariance
    cov_matrix = m.covariance
    output = ""
    for i in params_names:
        for j in params_names:
            key = (i,j)
            if key in cov_matrix:
                output += "%11.2e " % (cov_matrix[key])
        output += "\n"
    print output
    

    print "\nm.matrix()"
    print m.matrix(correlation=True)
    corr_matrix = m.matrix(correlation=True)
    output = ""
    for i in xrange(len(corr_matrix)):
        for j in xrange(len(corr_matrix[i])):
            output += "%9.2e " % (corr_matrix[i][j])
        output += "\n"
    print output
    
    print m.errors
    print m.values
    output = ""
    for k,v in number_of_events.iteritems():
        #print k,v
        nvals = number_of_events[k]
        if 'num_' in k:
            output += "%-12s %8.2f +/- %6.2f \t %8.2f +/- %6.2f\n" % (k,nvals['ndata'],nvals['ndata_err'],nvals['nacc_corr'],nvals['nacc_corr_err'])
    print output
    '''

    #print "Num org signal: ",num_org_sig
    #print "Num org background: ",num_org_bkg

    #print "Num acc signal: ",num_acc_sig
    #print "Num acc background: ",num_acc_bkg

    print "Num data events used: ",ndata

    #plotmc[0] = raw_plotmc[0][raw_plotmc[0]>0.5]
    #plotmc[1] = raw_plotmc[1][raw_plotmc[0]>0.5]
    
    #plotmc[0] = np.array(mc[0])
    #plotmc[1] = np.array(mc[1])

    nplotbins = 50

    #nsig_ac = number_of_events['num_sig']['nacc_corr']
    #nbkg_ac = number_of_events['num_bkg']['nacc_corr'] # - 100
    #nsig_ac = number_of_events['num_sig']['ndata']
    #nbkg_ac = number_of_events['num_bkg']['ndata'] # - 100
    #nsig_ac = number_of_events['num_sig']['pct']
    #nbkg_ac = number_of_events['num_bkg']['pct'] # - 100
    #nsig_ac =  635.0
    #nbkg_ac = 1818.0
    #nsig_ac =  335.0
    #nbkg_ac = 2118.0
    #nsig_ac = ndata*sig_frac
    #nbkg_ac = ndata*bkg_frac

    #print "nums: ",nsig_ac,nbkg_ac

    ######################
    # Plot total
    ######################
    '''
    print "total"
    acc_mc_weights = fitfunc(plotmc,m.args,params_names,params_dict)
    plot = plot_solution(plotmc[0],acc_mc_weights,nbins=nplotbins,range=ranges[0],axes_bin_width=bin_widths[0],ndata=ndata,axes=ax00,fmt='b-',linewidth=2)
    plot = plot_solution(plotmc[1],acc_mc_weights,nbins=nplotbins,range=ranges[1],axes_bin_width=bin_widths[1],ndata=ndata,axes=ax01,fmt='b-',linewidth=2)

    ######################
    # Plot signal
    ######################
    print "signal"
    temp_vals = list(m.args)
    temp_vals[params_names.index('num_bkg')] = 0.0
    acc_mc_weights = fitfunc(plotmc,temp_vals,params_names,params_dict)
    plot = plot_solution(plotmc[0],acc_mc_weights,nbins=nplotbins,range=ranges[0],axes_bin_width=bin_widths[0],ndata=nsig_ac,axes=ax00,fmt='r-',linewidth=2)
    plot = plot_solution(plotmc[1],acc_mc_weights,nbins=nplotbins,range=ranges[1],axes_bin_width=bin_widths[1],ndata=nsig_ac,axes=ax01,fmt='r-',linewidth=2)

    ######################
    # Plot background
    ######################
    print "background"
    temp_vals = list(m.args)
    temp_vals[params_names.index('num_sig')] = 0.0
    acc_mc_weights = fitfunc(plotmc,temp_vals,params_names,params_dict)
    plot = plot_solution(plotmc[0],acc_mc_weights,nbins=nplotbins,range=ranges[0],axes_bin_width=bin_widths[0],ndata=nbkg_ac,axes=ax00,fmt='g-',linewidth=2)
    plot = plot_solution(plotmc[1],acc_mc_weights,nbins=nplotbins,range=ranges[1],axes_bin_width=bin_widths[1],ndata=nbkg_ac,axes=ax01,fmt='g-',linewidth=2)
    '''

    print "\nnum_surv_signal: ",num_surv_signal

    #'''
    ############################################################################
    # Plot on the x-projection
    ############################################################################
    ytot = np.zeros(1000)
    xpts = np.linspace(ranges[0][0],ranges[0][1],1000)
    eff = sigmoid(xpts,threshold,sigmoid_sigma,max_val)

    # Sig
    gauss = stats.norm(loc=values['mean'],scale=values['sigma'])
    ypts = gauss.pdf(xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[0],scale=values['num_sig'],fmt='y--',axes=ax00,efficiency=eff)
    ytot += y

    # Bkg
    #bkg_exp = stats.expon(loc=0.0,scale=values['exp_bkg_x'])
    #ypts = bkg_exp.pdf(xpts)
    ypts = np.exp(-values['exp_bkg_x']*xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[0],scale=values['num_bkg'],fmt='g--',axes=ax00,efficiency=eff)
    ytot += y

    ax00.plot(xpts,ytot,'b',linewidth=3)

    ############################################################################
    # Plot on the y-projection
    ############################################################################
    #'''
    ytot = np.zeros(1000)
    xpts = np.linspace(ranges[1][0],ranges[1][1],1000)

    # Sig
    sig_exp = stats.expon(loc=0.0,scale=1.0)
    ypts = sig_exp.pdf(values['exp_sig_y']*xpts)

    y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[1],scale=values['num_sig'],fmt='y--',axes=ax01)
    ytot += y

    # Bkg
    ypts = np.ones(len(xpts))

    #y,plot = plot_pdf(xpts,ypts,bin_width=bin_widths[1],scale=values['num_bkg'],fmt='g--',axes=ax01)

    #print xpts,ypts
    totnorm = integrate.simps(ypts,x=xpts)
    norms = []
    for sr in subranges[1]:
        print sr[0],sr[1]
        xnorm = np.linspace(sr[0],sr[1],1000)
        ynorm = np.ones(len(xnorm))
        norm = integrate.simps(ynorm,x=xnorm)
        norms.append(norm)
        print "norms: ",totnorm,norm,totnorm/norm

    for norm,sr in zip(norms,subranges[1]):
        print sr[0],sr[1]
        xnorm = np.linspace(sr[0],sr[1],1000)
        ynorm = np.ones(len(xnorm))
        scale = norm/sum(norms)
        print "scale: ",scale,scale*values['num_bkg']
        y,plot = plot_pdf(xnorm,ynorm,bin_width=bin_widths[1],scale=scale*values['num_bkg'],fmt='g--',axes=ax01)
        print y
        ytot += y

    #ytot += y

    ax01.plot(xpts,ytot,'b',linewidth=3)
    #'''

    ############################################################################
    #'''

    plt.show()

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
    #lch.hist_err(mc,bins=108,range=(lo,hi))


################################################################################
################################################################################
if __name__=="__main__":
    main()
