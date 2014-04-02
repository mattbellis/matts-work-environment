from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import sys

minlh = 14864.9652913

infile = open(sys.argv[1])

xsec,mass,lh = np.loadtxt(sys.argv[1],usecols=(0,1,2),unpack=True)

lh -= minlh

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)

for m in (6,8,10,15,20,30):

    name = r"$M_{DM}$ = %d GeV/c$^2$" % (m)

    ax.plot(xsec[mass==m],lh[mass==m],'o',label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma_N$',fontsize=18)
    ax.set_ylabel(r'$\Delta \mathcal{L}$',fontsize=18)

    #ax2.plot(xsec[mass==m],np.exp(-lh[mass==m]),'o-',label=name)
    ax2.plot(xsec[mass==m],np.sqrt(2*lh[mass==m]),'o-',label=name)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\sigma_N$',fontsize=18)
    ax2.set_ylabel(r'$\sqrt{2\Delta \mathcal{L}}$',fontsize=18)


plt.legend()

plt.show() 
