import numpy as np
import matplotlib.pylab as plt

import scipy.constants as cnt

m = 32./1000/cnt.N_A
k = cnt.k
T0 = 300 # K
T1 = 800 # K
pi = np.pi

v = np.linspace(0,2000,1000)
P0 = np.sqrt((m/(2*pi*k*T0))**3)*4*pi*v**2*np.exp(-m*v*v/(2*k*T0))

P1 = np.sqrt((m/(2*pi*k*T1))**3)*4*pi*v**2*np.exp(-m*v*v/(2*k*T1))

plt.figure(figsize=(8,5))
plt.plot(v,P0,'-',linewidth=3,label='Maxwell-Boltzmann distribution (T=??? K)')
plt.plot(v,P1,'--',linewidth=3,label='Maxwell-Boltzmann distribution (T=??? K)')
plt.xlabel(r'$v$ (m/s)',fontsize=24)
plt.ylabel(r'$D(v)$',fontsize=24)
plt.legend(loc='upper right')

xticks = np.arange(0,2000,200)
yticks = np.arange(0,0.0025,0.00025)
plt.xticks(xticks)                                                       
plt.yticks(yticks)                                                       

plt.grid(which='major')

plt.tight_layout()

plt.savefig('mb_dist.png')



plt.show()
