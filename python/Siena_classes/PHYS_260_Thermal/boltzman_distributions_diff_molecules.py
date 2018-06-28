import numpy as np
import matplotlib.pylab as plt

import scipy.constants as cnt

m = [18./1000/cnt.N_A,54./1000/cnt.N_A]
k = cnt.k
T0 = 300 # K
pi = np.pi

v = np.linspace(0,2000,1000)
P0 = np.sqrt((m[0]/(2*pi*k*T0))**3)*4*pi*v**2*np.exp(-m[0]*v*v/(2*k*T0))

P1 = np.sqrt((m[1]/(2*pi*k*T0))**3)*4*pi*v**2*np.exp(-m[1]*v*v/(2*k*T0))

plt.figure(figsize=(8,5))
plt.plot(v,P0,'-',linewidth=3,label='M-B distribution for ??? gas')
plt.plot(v,P1,'--',linewidth=3,label='M-B distribution for ??? gas')
plt.xlabel(r'$v$ (m/s)',fontsize=24)
plt.ylabel(r'$D(v)$',fontsize=24)
plt.legend(loc='upper right',fontsize=18)

xticks = np.arange(0,2000,200)
yticks = np.arange(0,0.0025,0.00025)
plt.xticks(xticks)                                                       
plt.yticks(yticks)                                                       

plt.grid(which='major')

plt.tight_layout()

plt.savefig('mb_dist_diff_atoms.png')



plt.show()
