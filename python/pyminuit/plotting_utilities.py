import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from scipy.interpolate import spline
from scipy.interpolate import UnivariateSpline

from RTMinuit import *

################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf(x,ypts,bin_width=1.0,scale=1.0,efficiency=1.0,axes=None,fmt='-'):

    # Normalize to 1.0
    y = np.array(ypts)
    y *= efficiency
    normalization = integrate.simps(y,x=x)
    y /= normalization

    #print "exp int: ",integrate.simps(y,x=x)
    y *= (scale*bin_width)
    #print "bin_width: ",bin_width

    if axes==None:
        axes=plt.gca()

    plot = axes.plot(x,y,fmt,linewidth=2)
    #ytot += y
    #ax0.plot(x,ytot,'b',linewidth=3)

    return y,plot


################################################################################
# Plotting solution
################################################################################
def plot_solution(events,weights,nbins=100,range=(0,1),axes_bin_width=None,ndata=1.0,axes=None,fmt='-',linewidth=1):

    hist,bin_edges = np.histogram(events,bins=nbins,range=range,weights=weights,density=True)

    normalization = hist.sum()
     
    plot_bin_width = (bin_edges[1]-bin_edges[0])/2.0
    bin_centers = bin_edges[0:-1]+plot_bin_width
    #print "bin_centers: ",bin_centers

    if axes==None:
        axes=plt.gca()

    scaling_factor = 1.0
    if axes_bin_width!=None:
        scaling_factor = axes_bin_width/plot_bin_width
        #scaling_factor = 1.0

    #yvals = ndata*plot_bin_width*(scaling_factor)*hist/normalization
    yvals = ndata*plot_bin_width*(scaling_factor)*hist #/normalization

    print "integral: ",integrate.simps(hist,x=bin_centers)
    print "sum: ",yvals.sum()
    print "sum: ",hist.sum()

    xnew = np.linspace(bin_edges[0]-plot_bin_width,bin_edges[-1]+plot_bin_width,300)
    #ysmooth = spline(bin_centers,yvals,xnew)
    s = UnivariateSpline(bin_centers,yvals,s=1.0,k=2)
    ysmooth = s(xnew)

    #print "integral: ",integrate.simps(ysmooth,x=xnew)
    #print "sum: ",ysmooth.sum()

    plot = axes.plot(xnew,ysmooth,fmt,linewidth=linewidth)
    #plot = axes.plot(bin_centers,yvals,fmt,linewidth=linewidth)

    return plot
