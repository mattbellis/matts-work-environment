import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

x0 = stats.expon.rvs(size=10000,loc=0,scale=2.0)
x1 = stats.expon.rvs(size=10000,loc=0,scale=4.0)

y0 = stats.norm.rvs(loc=1 + x0/10.0,scale=0.5)
y1 = stats.norm.rvs(loc=4.5 + x1/10.0,scale=0.5)

plt.figure()
plt.hist(x0,bins=50,alpha=0.2,range=(0,20))
plt.hist(x1,bins=50,alpha=0.2,range=(0,20))

plt.figure()
plt.hist(y0,bins=50,alpha=0.2)
plt.hist(y1,bins=50,alpha=0.2)
plt.xlim(-5,12)

xtot = np.append(x0,x1)
ytot = np.append(y0,y1)

#'''
w0 = stats.norm.pdf(ytot,loc=1+xtot/10,scale=0.5)
w1 = stats.norm.pdf(ytot,loc=4.5+xtot/10,scale=0.5)

w0 /= (w0+w1)
w1 /= (w0+w1)

plt.figure()
plt.hist(xtot,bins=50,weights=w0,alpha=0.2,range=(0,20))
plt.hist(xtot,bins=50,weights=w1,alpha=0.2,range=(0,20))
plt.xlim(0,20)

#'''
plt.show()
