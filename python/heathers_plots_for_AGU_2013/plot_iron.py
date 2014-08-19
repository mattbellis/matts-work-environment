import numpy as np
from scipy import optimize         # simplified syntax
import matplotlib.pyplot as plt    # pylab != pyplot

import sys

# `unpack` lets you split the columns immediately:
filename = sys.argv[1]
y,y = np.loadtxt(filename, dtype=str, delimiter=',', usecols=(3,3), skiprows=1, unpack=True)

y = y.astype('float')
y0 = y[0:89]
y1 = y[90:191]
y2 = y[199:304]


print y0
npts0 = len(y0)
npts1 = len(y1)
npts2 = len(y2)

l0 = 0.050190476*np.ones(npts2)
l1 = 0.030190476*np.ones(npts2)
#l2 = 0.030190476*np.ones(npts2)
l2 = 0.035595*np.ones(npts2)
l3 = 0.035595*np.ones(npts2)

dx = 20.0

x0 = np.linspace(0,dx*npts0,npts0)
x1 = np.linspace(0,dx*npts1,npts1)
x2 = np.linspace(0,dx*npts2,npts2)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)

plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)

plt.plot(x0,y0,'o',label=r'1400$^\circ$ C - 60 hours (SGB-7)',markersize=8,color='k')
plt.plot(x1,y1,'v',label=r'1400$^\circ$ C - 24 hours',markersize=8,color='r')
plt.plot(x2,y2,'*',label=r'Untreated Mo foil',markersize=8,color='lightgreen')

plt.plot(x2,l0,'--',label='Detection limit (99% CI)',linewidth=3,color='r')
plt.plot(x2,l2,'-',label='Average measured value of SGB-7',linewidth=3,color='b')
plt.plot(x0,y0,'o',markersize=8,color='k')

plt.xlabel(r'Distance ($\mu$m)',fontsize=24)
plt.ylabel(r'Concentration Fe (wt %)',fontsize=24)

plt.xticks(fontsize=14) #, weight='bold')
plt.yticks(fontsize=14) #, weight='bold')

plt.ylim(0.0,.10)
plt.xlim(0.0,2500)

plt.legend(fontsize=12)

fig.savefig('iron_sdb7.jpg',transparent=False,frameon=True,facecolor='lightgrey')

#plt.show()
