import numpy as np
from scipy import optimize         # simplified syntax
import matplotlib.pyplot as plt    # pylab != pyplot

import sys

# `unpack` lets you split the columns immediately:
filename = 'char_diffusion.csv'
x,y0,y1,y2,y3,y4 = np.loadtxt(filename, dtype=float, delimiter=',', usecols=(0,1,2,3,4,5), skiprows=1, unpack=True)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)

plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)

D = 10e-13
y5 = np.sqrt(4*D*x*365.25*24*3600)

#plt.plot(x,y4,'-',label=r'D=10$^{-6}$',linewidth=4,color='k')
#plt.plot(x,y3,'--',label=r'D=10$^{-8}$',linewidth=4,color='red')
#plt.plot(x,y2,'-.',label=r'D=10$^{-9}$',linewidth=4,color='b')
#plt.plot(x,y1,':',label=r'D=10$^{-10}$',linewidth=4,color='green')
#plt.plot(x,y0,'-',label=r'D=10$^{-11}$',linewidth=4,color='orange')
#plt.plot(x,y5,'--',label=r'D=10$^{-13}$',linewidth=4,color='cyan')

plt.plot(x,y4,'-',label=r'$100000\times 10^{-11}$',linewidth=4,color='k')
plt.plot(x,y3,'--',label=r'$1000\times 10^{-11}$',linewidth=4,color='red')
plt.plot(x,y2,'-.',label=r'$100\times 10^{-11}$',linewidth=4,color='b')
plt.plot(x,y1,':',label=r'$10\times 10^{-11}$',linewidth=4,color='green')
plt.plot(x,y0,'-',label=r'$1\times 10^{-11}$',linewidth=4,color='orange')
plt.plot(x,y5,'--',label=r'$0.01\times 10^{-11}$',linewidth=4,color='cyan')


plt.xlabel(r'Time (years)',fontsize=24)
plt.ylabel(r'Distance (meters)',fontsize=24)

plt.yscale('log')

plt.xticks(fontsize=14) #, weight='bold')
plt.yticks(fontsize=14) #, weight='bold')

#plt.ylim(0.0,.10)
#plt.xlim(0.0,1.2e7)

plt.legend(fontsize=16,loc=9,bbox_to_anchor=(0.2, 0, 1, 1),title='Diffusion coefficient (D) [m$^2$/s]')

plt.savefig('figure_char_diff.jpg',transparent=False,frameon=True,facecolor='lightgrey')

plt.show()
