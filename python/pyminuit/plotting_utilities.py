import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from scipy.interpolate import spline
from scipy.interpolate import UnivariateSpline

#from RTMinuit import *
#from RTMinuit import *

################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf_from_lambda(func,bin_width=1.0,scale=1.0,efficiency=1.0,axes=None,fmt='-',subranges=None):

    y = None
    plot = None
    srxs = None

    if axes==None:
        axes=plt.gca()

    if subranges!=None:

        srxs = []
        tot_srys = []
        for sr in subranges:
            srxs.append(np.linspace(sr[0],sr[1],1000))
            tot_srys.append(np.zeros(1000))

        totnorm = 0.0
        srnorms = []
        y = []
        plot = []
        for srx,sr in zip(srxs,subranges):
            sry = func(srx)
            norm = integrate.simps(sry,x=srx)
            srnorms.append(norm)
            totnorm += norm

        for tot_sry,norm,srx,sr in zip(tot_srys,srnorms,srxs,subranges):
            norm /= totnorm

            ypts = func(srx)
            #print "norm*scale: ",norm*scale
            ytemp,plottemp = plot_pdf(srx,ypts,bin_width=bin_width,scale=norm*scale,fmt=fmt,axes=axes)
            y.append(ytemp)
            plot.append(plottemp)
            #tot_sry += y


    return y,plot,srxs
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
