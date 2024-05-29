import numpy as np
import matplotlib.pylab as plt

E = np.linspace(0,1000)
SA = 15*E 
SB = 10*E 


plt.figure(figsize=(10,5))
plt.plot(E,SA,'k--',linewidth=3,label='Object A')
plt.plot(E,SB,'k-',linewidth=2,label='Object B')
plt.xlabel('Energy in object (A or B) [Joules]',fontsize=24)
plt.ylabel('Entropy (units of k)',fontsize=24)

plt.legend(loc='upper left',fontsize=24)
plt.tight_layout()

#plt.savefig('SvsE_myplot.png')
plt.savefig('SvsE_myplot_v2.png')
plt.show()

