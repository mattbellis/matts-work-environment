import numpy as np
import matplotlib.pylab as plt
import cogent_utilities as cogent_utilities

from scipy import integrate
from cogent_utilities import *

x = np.linspace(0.5,3.2,1000)

for mu in [0.7,1.1,2.0,3.0]:
    s=0.001; y = np.exp(-((x-mu)**2)/(2*s*s))
    plt.plot(x,y,label='original')
    yc,xc = cogent_convolve(x,y)

    print integrate.simps(y,x=x)
    print integrate.simps(yc,x=x)

    plt.plot(x,yc,label='convolved')

    plt.legend()
 
plt.show()
