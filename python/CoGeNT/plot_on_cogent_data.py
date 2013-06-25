import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.integrate as integrate

from cogent_utilities import *
from fitting_utilities import *
from plotting_utilities import *

import lichen.lichen as lch

#import minuit

pi = np.pi

################################################################################
# Read in the CoGeNT data
################################################################################
def main():

    infile = open('data/before_fire_LG.dat')
    content = np.array(infile.read().split()).astype('float')
    ndata = len(content)/2
    index = np.arange(0,ndata*2,2)
    times = content[index]
    index = np.arange(1,ndata*2+1,2)
    amplitudes = content[index]
    energies = amp_to_energy(amplitudes,0)

    lo = 0.5
    hi = 3.2
    nbins = 108
    bin_width = (hi-lo)/nbins
    print bin_width

    fig0 = plt.figure(figsize=(10,9),dpi=100)
    ax0 = fig0.add_subplot(2,1,1)
    ax0.set_xlim(lo,hi)

    lch.hist_err(energies,bins=nbins,range=(lo,hi),axes=ax0)

    x = np.linspace(lo,hi,1000)

    ############################################################################
    # Get the efficiency function
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241
    efficiency = sigmoid(x,threshold,sigmoid_sigma,max_val)
    #fig1 = plt.figure()
    ax1 = fig0.add_subplot(2,1,2) 
    ax1.plot(x,efficiency,'r--',linewidth=2)
    ax1.set_xlim(lo,hi)
    ax1.set_ylim(0.0,1.0)

    #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data([[1,68],[75,102],[108,306],[309,459],[551,917]])
    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data([[1,68],[75,102],[108,306],[309,459]])
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
    yorg = surf_expon.pdf(3.3*x)
    y,surf_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=575.0,fmt='y-',axes=ax0,efficiency=efficiency)
    ytot += y

    ############################################################################
    # Flat term
    ############################################################################
    yorg = np.ones(len(x))
    y,flat_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=1060.0,fmt='m-',axes=ax0,efficiency=efficiency)
    ytot += y
    

    ############################################################################
    # WIMP-like term
    ############################################################################
    wimp_expon = stats.expon(scale=1.0)
    yorg = wimp_expon.pdf(2.3*x)
    y,wimp_plot = plot_pdf(x,yorg,bin_width=bin_width,scale=330.0,fmt='g-',axes=ax0,efficiency=efficiency)
    ytot += y
    
    ############################################################################
    # Total-like term
    ############################################################################
    ax0.plot(x,ytot,'b',linewidth=3)
    



    #data = [events,deltat_mc]
    #m = minuit.Minuit(pdfs.extended_maximum_likelihood_function_minuit,p=p0)
    #print m.values
    #m.migrad()



    plt.show()

################################################################################
################################################################################
if __name__=="__main__":
    main()
