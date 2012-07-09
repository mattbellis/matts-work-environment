import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datetime import datetime,timedelta

import scipy.integrate as integrate

from cogent_utilities import *
from cogent_pdfs import *
from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

import minuit

pi = np.pi
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

np.random.seed(200)

################################################################################
# Read in the CoGeNT data
################################################################################
def main():

    ############################################################################
    # Read in the data
    ############################################################################
    infile = open('data/before_fire_LG.dat')
    content = np.array(infile.read().split()).astype('float')
    ndata = len(content)/2
    index = np.arange(0,ndata*2,2)
    tseconds = content[index]
    tdays = (tseconds-first_event)/(24.0*3600.0) + 1.0
    index = np.arange(1,ndata*2+1,2)
    amplitudes = content[index]
    energies = amp_to_energy(amplitudes,0)

    #print tdays
    '''
    output = ""
    i = 0
    for e,t in zip(energies,tdays):
        if e<3.3:
            output += "%7.2f " % (t)
            i+=1
        if i==10:
            print output 
            output = ""
            i=0
    '''

    data = [np.array(energies),np.array(tdays)]

    print "data before range cuts: ",len(data[0]),len(data[1])

    ############################################################################
    # Declare the ranges.
    ############################################################################
    ranges = [[0.5,3.2],[1.0,459.0]]
    #dead_days = [[68,74], [102,107],[306,308]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
    nbins = [108,15]
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n

    # Cut events out that fall outside the range.
    data = cut_events_outside_range(data,ranges)
    data = cut_events_outside_subrange(data,subranges[1],data_index=1)

    print "data after range cuts: ",len(data[0]),len(data[1])

    '''
    output = ""
    i = 0
    for e,t in zip(data[0],data[1]):
        if e<3.3:
            output += "%7.2f " % (t)
            i+=1
        if i==10:
            print output 
            output = ""
            i=0
    '''

    nevents = float(len(data[0]))

    # Plot the data
    fig0 = plt.figure(figsize=(12,4),dpi=100)
    ax0 = fig0.add_subplot(1,2,1)
    ax1 = fig0.add_subplot(1,2,2)

    fig0.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    ax0.set_xlim(ranges[0])
    ax0.set_ylim(0.0,92.0)
    ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
    ax0.set_ylabel("Events/0.025 keVee",fontsize=12)

    ax1.set_xlim(ranges[1])
    ax1.set_ylim(0.0,220.0)
    ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
    ax1.set_ylabel("Event/30 days",fontsize=12)

    lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)
    # Do an acceptance correction
    tbwidth = (ranges[1][1]-ranges[1][0])/float(nbins[1])
    acc_corr = np.zeros(len(ypts))
    acc_corr[2] = tbwidth/(tbwidth-7.0)
    acc_corr[3] = tbwidth/(tbwidth-6.0)
    acc_corr[10] = tbwidth/(tbwidth-3.0)
    ax1.errorbar(xpts, acc_corr*ypts,xerr=xpts_err,yerr=acc_corr*ypts_err,fmt='o', \
                        color='red',ecolor='red',markersize=2,barsabove=False,capsize=0)

    # For efficiency
    fig1 = plt.figure(figsize=(12,4),dpi=100)
    ax10 = fig1.add_subplot(1,2,1)
    ax11 = fig1.add_subplot(1,2,2)

    fig1.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    ############################################################################
    # Gen some MC
    ############################################################################
    #nmcraw = 20000
    #mcraw = gen_mc(nmcraw,ranges)

    ############################################################################
    # Run the MC through the efficiency.
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241

    efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)
    #efficiency = lambda x: 1.0

    #mcacc = cogent_efficiency(mcraw,threshold,sigmoid_sigma,max_val)

    #exit()
    
    ############################################################################
    # Fit
    ############################################################################
    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(458)

    # Might need this to take care of efficiency.
    #num_decays_in_dataset *= 0.87

    tot_lshells = num_decays_in_dataset.sum()
    print "Total number of lshells in dataset: ",tot_lshells

    ############################################################################
    # Fit
    ############################################################################

    # Declare the parameters
    params_dict = {}
    #params_dict['flag'] = {'fix':True,'start_val':0}
    params_dict['flag'] = {'fix':True,'start_val':1} # Normalized version
    params_dict['var_e'] = {'fix':True,'start_val':0,'limits':(ranges[0][0],ranges[0][1])}
    params_dict['var_t'] = {'fix':True,'start_val':0,'limits':(ranges[1][0],ranges[1][1])}

    # L-shell parameters
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

    yearly_mod = 2*pi/365.0

    # Exponential term in energy
    params_dict['e_exp0'] = {'fix':False,'start_val':2.51,'limits':(0.0,10.0)}
    params_dict['e_exp1'] = {'fix':True,'start_val':3.36,'limits':(0.0,10.0)}
    #params_dict['num_exp0'] = {'fix':False,'start_val':296.0,'limits':(0.0,10000.0)}
    params_dict['num_exp0'] = {'fix':True,'start_val':1.0,'limits':(0.0,10000.0)}
    params_dict['num_exp1'] = {'fix':True,'start_val':575.0,'limits':(0.0,100000.0)}
    #params_dict['num_exp1'] = {'fix':True,'start_val':506.0,'limits':(0.0,100000.0)}
    params_dict['num_flat'] = {'fix':False,'start_val':1159.0,'limits':(0.0,100000.0)}

    #params_dict['wmod_freq'] = {'fix':True,'start_val':yearly_mod,'limits':(0.0,10000.0)}
    #params_dict['wmod_phase'] = {'fix':False,'start_val':0.00,'limits':(-2*pi,2*pi)}
    #params_dict['wmod_amp'] = {'fix':False,'start_val':0.20,'limits':(0.0,1.0)}
    #params_dict['wmod_offst'] = {'fix':True,'start_val':1.00,'limits':(0.0,10000.0)}
    params_dict['mDM'] = {'fix':False,'start_val':7.00,'limits':(0.0,10000.0)}

    params_names,kwd = dict2kwd(params_dict)

    #f = Minuit_FCN([data,mcacc],params_dict)
    f = Minuit_FCN([data],params_dict)

    m = minuit.Minuit(f,**kwd)

    #exit()

    # For maximum likelihood method.
    m.up = 0.5

    m.printMode = 0

    m.migrad()
    m.hesse()

    values = m.values # Dictionary

    m1 = m
    m2 = m

    '''
    print "starting contours..."
    plt.figure()
    for sig in [1.5,1.0]:
        contour_points = None
        print "----"
        print sig
        #print m.values
        if sig==1.0:
            contour_points = m1.contour('e_exp0','num_exp0',sig,40)
        else:
            contour_points = m2.contour('e_exp0','num_exp0',sig,40)
        print contour_points 
        cx = np.array([])
        cy = np.array([])
        if contour_points!=None and len(contour_points)>1:
            for p in contour_points:
                cx = np.append(cx,p[0])
                cy = np.append(cy,p[1])
            cx = np.append(cx,contour_points[0][0])
            cy = np.append(cy,contour_points[0][1])
        plt.plot(cx,cy)
    '''

    '''
    print "\nm.matrix()"
    print m.matrix(correlation=True)
    corr_matrix = m.matrix(correlation=True)
    output = ""
    for i in xrange(len(corr_matrix)):
        for j in xrange(len(corr_matrix[i])):
            output += "%9.2e " % (corr_matrix[i][j])
        output += "\n"
    print output
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
                #output += "%11.2e " % (cov_matrix[key])
                output += "%-12s %-12s %11.4f\n" % (i,j,cov_matrix[key])
        #output += "\n"
    print output
    '''

    print "Finished fit!!\n"
    print minuit_output(m)
    print "nentries: ",len(data[0])

    #exit()

    names = []
    for name in params_names:
        if 'num_' in name or 'ncalc' in name:
            names.append(name)

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

    eff = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    #eff = np.ones(1000)

    srxs = []
    tot_srys = []
    for sr in subranges[1]:
        srxs.append(np.linspace(sr[0],sr[1],1000))
        tot_srys.append(np.zeros(1000))

    ############################################################################
    # Exponential
    ############################################################################
    # Energy projections
    ypts = np.exp(-values['e_exp0']*expts)
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp0'],fmt='g-',axes=ax0,efficiency=eff)
    eytot += y

    # Time projections
    func = lambda x: np.ones(len(x))
    #func = lambda x: values['wmod_offst'] + values['wmod_amp']*np.cos(values['wmod_freq']*x+values['wmod_phase'])   
    srys,plot,srxs = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp0'],fmt='g-',axes=ax1,subranges=subranges[1])
    tot_srys = [tot + y for tot,y in zip(tot_srys,srys)]


    ############################################################################
    # Second exponential
    ############################################################################
    # Energy projections
    ypts = np.exp(-values['e_exp1']*expts)
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp1'],fmt='y-',axes=ax0,efficiency=eff)
    eytot += y

    # Time projections
    func = lambda x: np.ones(len(x))
    srys,plot,srxs = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_exp1'],fmt='y-',axes=ax1,subranges=subranges[1])
    tot_srys = [tot + y for tot,y in zip(tot_srys,srys)]
    #ypts = np.ones(len(txpts))
    #y,plot = plot_pdf(txpts,ypts,bin_width=bin_widths[1],scale=values['num_exp1'],fmt='y-',axes=ax1)
    #tytot += y

    ############################################################################
    # Flat
    ############################################################################
    # Energy projections
    ypts = np.ones(len(expts))
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=eff)
    eytot += y
    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='r-',axes=ax0)

    # Time projections
    #typts = np.ones(len(txpts))
    #y,plot = plot_pdf(txpts,typts,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1)
    #tytot += y
    func = lambda x: np.ones(len(x))
    srys,plot,srxs = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=values['num_flat'],fmt='m-',axes=ax1,subranges=subranges[1])
    tot_srys = [tot + y for tot,y in zip(tot_srys,srys)]

    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='r-',axes=ax0)
    # L-shell
    # Returns pdfs
    lshell_totx = np.zeros(1000)
    lshell_toty = np.zeros(1000)
    for m,s,n,dc in zip(means,sigmas,num_decays_in_dataset,decay_constants):
        gauss = stats.norm(loc=m,scale=s)
        eypts = gauss.pdf(expts)

        y,plot = plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=n,fmt='r--',axes=ax0,efficiency=eff)
        eytot += y
        lshell_totx += y

        # Time distribution
        #pdf_e = stats.expon(loc=0.0,scale=-1.0/dc)
        #typts = pdf_e.pdf(txpts)
        #typts = np.exp(dc*txpts)
        #y,plot = plot_pdf(txpts,typts,bin_width=bin_widths[1],scale=n,fmt='r--',axes=ax1)
        #lshell_toty += y

        #tytot += y
        func = lambda x: np.exp(dc*x)
        srys,plot,srxs = plot_pdf_from_lambda(func,bin_width=bin_widths[1],scale=n,fmt='r--',axes=ax1,subranges=subranges[1])
        tot_srys = [tot + y for tot,y in zip(tot_srys,srys)]
        lshell_toty = [tot + y for tot,y in zip(lshell_toty,srys)]


    ax0.plot(expts,lshell_totx,'r-',linewidth=2)

    ax0.plot(expts,eytot,'b',linewidth=3)
    #ax1.plot(txpts,tytot,'b',linewidth=3)

    # Total on y/t
    for x,y,lsh in zip(srxs,tot_srys,lshell_toty):
        ax1.plot(x,lsh,'r-',linewidth=2)
        ax1.plot(x,y,'b',linewidth=3)


    ############################################################################

    # Efficiency function
    efficiency = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    ax10.plot(expts,efficiency,'r--',linewidth=2)
    ax10.set_xlim(ranges[0][0],ranges[0][1])
    ax10.set_ylim(0.0,1.0)

    plt.show()

    exit()


################################################################################
################################################################################
if __name__=="__main__":
    main()
