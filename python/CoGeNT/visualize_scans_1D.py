from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from scipy.integrate import quad

import sys

#minlh = 14864.9652913
minlh = 14857.0676672

infile = open(sys.argv[1])

tag = "default"
if len(sys.argv)>2:
    tag = sys.argv[2]

xsec,mass,lh = np.loadtxt(sys.argv[1],usecols=(0,1,2),unpack=True)

lh -= minlh

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

#for m in (6,6):
#for m in (6,8,10,15,20,30):
for m in (10,15,20,30): # For stream

    name = r"$M_{DM}$ = %d GeV/c$^2$" % (m)

    ax.plot(xsec[mass==m],lh[mass==m],'o',label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma_N$',fontsize=24)
    ax.set_ylabel(r'$\Delta \mathcal{L}$',fontsize=24)

    ax2.plot(xsec[mass==m],np.sqrt(2*lh[mass==m]),'o-',label=name)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\sigma_N$',fontsize=24)
    ax2.set_ylabel(r'$\sqrt{2\Delta \mathcal{L}}$',fontsize=24)

    ax3.plot(xsec[mass==m],np.exp(-lh[mass==m]),'o-',label=name)
    ax3.set_xscale('log')
    #ax3.set_yscale('log')
    ax3.set_xlabel(r'$\sigma_N$',fontsize=24)
    ax3.set_ylabel(r'$\sqrt{2\Delta \mathcal{L}}$',fontsize=24)

    xpts = xsec[mass==m]
    ypts = np.exp(-lh[mass==m])

    integral = np.trapz(xpts,ypts)
    #print "--------------------------------------------------"
    #print m
    #print integral
    for i in range(1,len(ypts)):
        #print "xpts,ypts: ", xpts[-i],ypts[-i]
        integral90 = np.trapz(xpts[0:-i],ypts[0:-i])
        #print integral90,integral
        #print integral90/integral
        if (integral90/integral<0.9):
            print m,",",xsec[mass==m][-i]
            break


fig.subplots_adjust(bottom=0.15)
fig3.subplots_adjust(bottom=0.15)
plt.legend()

fig2.subplots_adjust(bottom=0.15)
ax2.legend()
name = "Plots/delta_log_lh_%s.png" % (tag)
fig2.savefig(name)



plt.show() 
