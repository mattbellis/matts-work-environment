import numpy as np
import matplotlib.pylab as plt
import cogent_utilities as cogent_utilities

from scipy import integrate
from cogent_utilities import *
import scipy.signal as signal
import scipy.stats as stats

x = np.linspace(0.5,1.5,1000)

#for mu in [0.7,1.1,2.0,3.0]:
for mu in [1.1]:
    #s=0.001
    s=0.1
    y = np.exp(-((x-mu)**2)/(2*s*s))
    plt.plot(x,y,label='original')
    #yc,xc = cogent_convolve(x,y)

    xpts = np.linspace(-5,5,1000)
    #xpts = np.linspace(0.5,1.5,100)
    sigma = 0.5
    #convolving_pts = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-((xpts-0.0)**2)/(2*sigma*sigma))
    convolving_term = stats.norm(0.0,sigma)
    convolving_pts = convolving_term.pdf(xpts)
    #convolving_pts = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-0.0)**2)/(2*sigma*sigma))

    #yc = signal.fftconvolve(y/y.sum(),convolving_pts,mode='same')
    yc = signal.fftconvolve(y/y.sum(),convolving_pts,mode='same')

    print integrate.simps(y,x=x)
    print integrate.simps(yc,x=x)

    plt.plot(x,yc,label='convolved')

    plt.legend()
 
plt.show()
