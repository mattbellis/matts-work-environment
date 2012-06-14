import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.integrate as integrate

from cogent_utilities import *
from fitting_utilities import *

import lichen.lichen as lch

import minuit

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

    fig0 = plt.figure(figsize=(10,5),dpi=100)
    ax0 = fig0.add_subplot(1,1,1)
    lo = 0.5
    hi = 3.2
    nbins = 100
    bin_width = (hi-lo)/nbins
    print bin_width
    #ax0.hist(energies,bins=nbins,range=(lo,hi))
    lch.hist_err(energies,bins=nbins,range=(lo,hi),axes=ax0)

    x = np.linspace(lo,hi,1000)

    ############################################################################
    # Get the efficiency function
    ############################################################################
    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241
    efficiency = sigmoid(x,threshold,sigmoid_sigma,max_val)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.plot(x,efficiency,linewidth=3)
    ax1.set_ylim(0.0,1.0)

    means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)
    #means = np.array([1.2977,1.1])
    #sigmas = np.array([0.077,0.077])
    #numbers = np.array([638,50])

    lshells = lshell_peaks(means,sigmas,num_decays_in_dataset)
    print lshells
    ytot = np.zeros(1000)
    print means
    for n,cp in zip(num_decays_in_dataset,lshells):
        tempy = cp.pdf(x)
        y = n*cp.pdf(x)*bin_width*efficiency
        print n,integrate.simps(tempy,x=x),integrate.simps(y,x=x)
        ytot += y
        ax0.plot(x,y,'r--',linewidth=2)
    ax0.plot(x,ytot,'r',linewidth=3)

    ############################################################################
    # Surface term
    ############################################################################
    surf_expon = stats.expon(scale=1.0)
    # Normalize to 1.0
    y = surf_expon.pdf(3.3*x)
    normalization = integrate.simps(y,x=x)
    y /= normalization
    print "exp int: ",integrate.simps(y,x=x)
    y *= (575.0*bin_width)*efficiency
    ax0.plot(x,y,'y',linewidth=2)
    ytot += y
    #ax0.plot(x,ytot,'b',linewidth=3)

    ############################################################################
    # Flat term
    ############################################################################
    y = np.ones(len(x))
    normalization = integrate.simps(y,x=x)
    y /= normalization
    print "exp flat: ",integrate.simps(y,x=x)
    y *= 1060.0*bin_width*efficiency
    ax0.plot(x,y,'m',linewidth=2)
    ytot += y
    #ax0.plot(x,ytot,'b',linewidth=3)

    ############################################################################
    # WIMP-like term
    ############################################################################
    wimp_expon = stats.expon(scale=1.0)
    # Normalize to 1.0
    y = wimp_expon.pdf(2.3*x)
    normalization = integrate.simps(y,x=x)
    y /= normalization
    print "exp wimp: ",integrate.simps(y,x=x)
    y *= (330.0*bin_width)*efficiency
    ax0.plot(x,y,'g',linewidth=2)
    ytot += y
    #ax0.plot(x,ytot,'b',linewidth=3)

    #print ytot
    #ytot *= eff_func
    #print ytot
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
