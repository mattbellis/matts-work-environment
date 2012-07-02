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

    print tdays
    data = [energies,tdays]

    ############################################################################
    # Declare the ranges.
    ############################################################################
    ranges = [[0.5,3.2],[0.0,458.0]]
    #dead_days = [[68,74], [102,107],[306,308]]
    subranges = [[],[1,68],[[74,102],[107,306],[308,458]]]
    nbins = [108,15]
    bin_widths = np.ones(len(ranges))
    for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
        bin_widths[i] = (r[1]-r[0])/n

    # Cut events out that fall outside the range.
    data = cut_events_outside_range(data,ranges)

    nevents = float(len(data[0]))

    # Plot the data
    fig0 = plt.figure(figsize=(12,4),dpi=100)
    ax0 = fig0.add_subplot(1,2,1)
    ax1 = fig0.add_subplot(1,2,2)

    fig0.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    ax0.set_xlim(ranges[0])
    ax0.set_ylim(0.0,100.0)
    ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
    ax0.set_ylabel("Events/0.025 keVee",fontsize=12)

    ax1.set_xlim(ranges[1])
    ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
    ax1.set_ylabel("Event/30 days",fontsize=12)

    lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)
    lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)

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
    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)

    # Might need this to take care of efficiency.
    #num_decays_in_dataset *= 0.87

    tot_lshells = num_decays_in_dataset.sum()

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

    # Exponential term in energy
    params_dict['e_exp0'] = {'fix':False,'start_val':2.51,'limits':(0.0,10.0)}
    params_dict['e_exp1'] = {'fix':True,'start_val':3.36,'limits':(0.0,10.0)}
    params_dict['num_exp0'] = {'fix':False,'start_val':296.0,'limits':(100.0,1000.0)}
    params_dict['num_exp1'] = {'fix':True,'start_val':575.0,'limits':(0.0,100000.0)}
    params_dict['num_flat'] = {'fix':False,'start_val':1159.0,'limits':(0.0,100000.0)}

    params_names,kwd = dict2kwd(params_dict)

    #f = Minuit_FCN([data,mcacc],params_dict)
    f = Minuit_FCN([data],params_dict)

    m = minuit.Minuit(f,**kwd)

    #exit()

    # For maximum likelihood method.
    m.up = 0.5

    m.printMode = 1

    m.migrad()

    values = m.values # Dictionary

    print "Finished fit!!\n"
    print minuit_output(m)

    #print m.fixed

    #exit()

    #acc_integral_tot = fitfunc(mcacc,m.args,params_names,params_dict).sum()
    #print "acc_integral_tot: ",acc_integral_tot

    nfracs = []
    #names = ['num_exp0','num_flat']
    #names = ['num_exp0','num_exp1','num_flat']
    names = []
    for name in params_names:
        if 'num_' in name or 'ncalc' in name:
            names.append(name)

    '''
    for name in names:
        temp_vals = list(m.args)
        # Set all but name to 0.0
        for zero_name in names:
            if name!= zero_name:
                temp_vals[params_names.index(zero_name)] = 0.0
                #print "zeroing out",zero_name
        acc_integral_temp = fitfunc(mcacc,temp_vals,params_names,params_dict).sum()
        print "acc_integral_temp: ",acc_integral_temp
        frac = acc_integral_temp/acc_integral_tot
        print "frac: ",name,frac
        nfracs.append(frac)

    npdfs = {}
    totls = 0.0
    for n,f in zip(names,nfracs):
        print "%-12s: %f" % (n,(f*nevents)) 
        npdfs[n] = (f*nevents)
        if 'ncalc' in n:
            totls += f*nevents
    print "%-12s: %f" % ("L-shells",tot_lshells) 
    print "%-12s: %f" % ("totls",totls) 
    print "%-12s: %f" % ("tot",nevents) 
    print "%-12s: %f" % ("nmcacc",len(mcacc[0])) 
    '''

    ############################################################################
    # Plot the solutions
    ############################################################################
    ############################################################################
    # Plot on the x-projection
    ############################################################################
    expts = np.linspace(ranges[0][0],ranges[0][1],1000)
    txpts = np.linspace(ranges[1][0],ranges[1][1],1000)

    ytot = np.zeros(1000)
    expts = np.linspace(ranges[0][0],ranges[0][1],1000)
    eff = sigmoid(expts,threshold,sigmoid_sigma,max_val)
    #eff = np.ones(1000)

    # Exponential
    ypts = np.exp(-values['e_exp0']*expts)
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp0'],fmt='g-',axes=ax0,efficiency=eff)
    ytot += y

    # Second exponential
    ypts = np.exp(-values['e_exp1']*expts)
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_exp1'],fmt='y-',axes=ax0,efficiency=eff)
    ytot += y

    # Flat
    ypts = np.ones(len(expts))
    y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0,efficiency=eff)
    ytot += y
    #y,plot = plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='r-',axes=ax0)

    # L-shell
    # Returns pdfs
    lshell_totx = np.zeros(1000)
    lshell_toty = np.zeros(1000)
    for m,s,n,dc in zip(means,sigmas,num_decays_in_dataset,decay_constants):
        gauss = stats.norm(loc=m,scale=s)
        eypts = gauss.pdf(expts)

        y,plot = plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=n,fmt='r--',axes=ax0,efficiency=eff)
        ytot += y
        lshell_totx += y

        # Time distribution
        #pdf_e = stats.expon(loc=0.0,scale=-1.0/dc)
        #typts = pdf_e.pdf(txpts)
        typts = np.exp(dc*txpts)

        y,plot = plot_pdf(txpts,typts,bin_width=bin_widths[1],scale=n,fmt='r--',axes=ax1)
        lshell_toty += y

    ax0.plot(expts,lshell_totx,'r-',linewidth=2)
    ax1.plot(txpts,lshell_toty,'r-',linewidth=2)

    ax0.plot(expts,ytot,'b',linewidth=3)


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
