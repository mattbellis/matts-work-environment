import numpy as np
import matplotlib.pylab as plt
import cogent_utilities as cogent_utilities

from scipy import integrate
from cogent_utilities import *
import scipy.signal as signal

import scipy.stats as stats

x = np.linspace(0.9,1.3,1000)

#for mu in [0.7,1.1,2.0,3.0]:
for mu in [1.1]:
    s=0.001
    y = np.exp(-((x-mu)**2)/(2*s*s))
    plt.plot(x,y/y.sum(),label='original')
    #yc,xc = cogent_convolve(x,y)

    sigma = 0.04
    window = 3.0*sigma
    yc = np.array([])
    tau = 3.0
    for pt in x:
        # Get the x points used for the convolution.
        sigma = 0.002*pt
        window = 3.0*sigma
        temp_pts = np.linspace(pt-window,pt+window,1000)

        val = 0.0
        convolving_term = stats.norm(0.0,sigma)
        #val += (np.exp(-abs(temp_pts)*tau)*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()
        val += (np.exp((-((temp_pts-mu)**2)/(2*s*s)))*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()

        yc = np.append(yc,val)


        #yc = signal.convolve(y/y.sum(),convolving_pts,mode='same')

    yc /= yc.sum()

    print integrate.simps(y,x=x)
    print integrate.simps(yc,x=x)

    plt.plot(x,yc,label='convolved')

    plt.legend()
 
plt.show()
