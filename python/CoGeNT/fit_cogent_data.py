import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from cogent_utilities import *
from fitting_utilities import *

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

    plt.figure()
    plt.hist(energies,bins=100,range=(0.5,3.2))

    means = np.array([1.2,1.1])
    sigmas = np.array([0.1,0.1])
    numbers = np.array([20,50])

    lshells = cosmogenic_peaks(means,sigmas,numbers)
    x = np.linspace(0.5,3.2,1000)
    print lshells
    for n,cp in zip(numbers,lshells):
        y = n*cp.pdf(x)
        plt.plot(x,y)

    #data = [events,deltat_mc]
    #m = minuit.Minuit(pdfs.extended_maximum_likelihood_function_minuit,p=p0)
    #print m.values
    #m.migrad()



    plt.show()

################################################################################
################################################################################
if __name__=="__main__":
    main()
