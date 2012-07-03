import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from RTMinuit import *

################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf(x,ypts,bin_width=1.0,scale=1.0,efficiency=1.0,axes=None,fmt='-'):

    y = np.array(ypts)
    y *= efficiency

    # Normalize to 1.0
    normalization = integrate.simps(y,x=x)
    y /= normalization

    #print "exp int: ",integrate.simps(y,x=x)
    #y *= (scale*bin_width)*efficiency
    y *= (scale*bin_width)

    if axes==None:
        axes=plt.gca()

    plot = axes.plot(x,y,fmt,linewidth=2)
    #ytot += y
    #ax0.plot(x,ytot,'b',linewidth=3)

    return y,plot


