import numpy as np
import matplotlib.pylab as plt
import cogent_utilities as cogent_utilities

from scipy import integrate
from cogent_utilities import *
import scipy.signal as signal

import scipy.stats as stats

#x = np.linspace(0.9,1.3,1000)
x = np.linspace(-2,3.2,1000)

#for mu in [1.1]:
for mu in [2.5]:
    
    # This should be normalized now. (?)
    y = (mu)*np.exp(-mu*x)

    y[x<=0] = 0

    ynorm = integrate.simps(y,x=x)

    plt.plot(x,y/ynorm,label='original',linewidth=3)
    yc,xc = cogent_convolve(x,y)

    #yc /= yc.sum()

    ycnorm = integrate.simps(yc,x=x)

    plt.plot(x,yc/ycnorm,label='convolved',linewidth=3)

    plt.legend()
 
plt.show()
