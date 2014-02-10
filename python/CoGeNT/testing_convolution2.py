import numpy as np
import matplotlib.pylab as plt
import cogent_utilities as cogent_utilities

from scipy import integrate
from cogent_utilities import *
import scipy.signal as signal

import scipy.stats as stats

#x = np.linspace(0.9,1.3,1000)
x = np.linspace(0.0,4.2,1000)

#for mu in [1.1]:
#for mu in [0.7,1.1,2.0,3.0]:
for mu in [0.5,0.6,0.6,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
    s = 0.005
    y = (1.0/(np.sqrt(2*np.pi)*s))*np.exp(-((x-mu)**2)/(2*s*s))
    plt.plot(x,y,label='original')
    yc,xc = cogent_convolve(x,y)

    print integrate.simps(y,x=x)
    print integrate.simps(yc,x=x)

    plt.plot(x,yc,label='convolved')

    plt.legend()
 
plt.show()
