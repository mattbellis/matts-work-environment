import numpy as np
import matplotlib.pylab as plt

import scipy.constants as cnt

plt.figure(figsize=(8,5))
plt.plot([2,10,10,2],[1e5,3e5,1e5,1e5],'-',linewidth=3)#,label='Maxwell-Boltzmann distribution (T=??? K)')
plt.xlabel(r'$V$ [m$^3$]',fontsize=24)
plt.ylabel(r'$P$ [Pa]',fontsize=24)
#plt.legend(loc='upper right')

plt.xlim(0,12)
plt.ylim(0,4e5)

#xticks = np.arange(0,2000,200)
#yticks = np.arange(0,0.0025,0.00025)
#plt.xticks(xticks)                                                       
#plt.yticks(yticks)                                                       

plt.grid(which='major')

plt.tight_layout()

plt.savefig('engine_cycle.png')



plt.show()
