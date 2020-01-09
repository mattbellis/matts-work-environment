import matplotlib.pylab as plt
import numpy as np

import scipy.stats as stats
import lichen as lch


################################################################################
# Basic example
################################################################################
x = np.linspace(100,750,1000)

pdf1 = stats.expon(loc=100,scale=200.0)
bkg = pdf1.rvs(size=1000000)

pdf2 = stats.norm(loc=400,scale=20.0)
sig = pdf2.rvs(size=1000000)

plt.figure(figsize=(4,6))
plt.subplot(2,1,1)
#plt.hist([bkg,sig],bins=100,range=(100,750),stacked=True)
#plt.hist(bkg,bins=100,range=(100,750),label='MC background')
plt.plot(x,pdf1.pdf(x),linewidth=4,label='MC background')
plt.xlim(50,800)
plt.ylabel('Entries')
plt.xlabel(r'Invariant mass [GeV/c$^2$]')
plt.legend()

plt.subplot(2,1,2)
#plt.hist(sig,bins=100,range=(100,750),color='red',label='MC signal')
plt.plot(x,pdf2.pdf(x),linewidth=4,color='red',label='MC signal')
plt.xlim(50,800)
plt.ylabel('Entries')
plt.xlabel(r'Invariant mass [GeV/c$^2$]')
plt.legend()
plt.tight_layout()
plt.savefig('example1_MC.png')

data = bkg[0:10000].tolist() + sig[0:1000].tolist()

plt.figure()
lch.hist(data,bins=100,range=(100,750),label='Data')
plt.ylabel('Entries',fontsize=18)
plt.xlabel(r'Invariant mass [GeV/c$^2$]',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig('example1_data.png')
################################################################################


plt.show()
