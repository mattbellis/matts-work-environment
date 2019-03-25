import matplotlib.pylab as plt
import numpy as np

import scipy.stats as stats
import scipy.signal as sig

x = np.convolve([1, 2, 3], [0, 1, 0.5])
print(x)

x = np.linspace(0,600,1000)
y = np.exp(-x/150)

xres = np.linspace(-10,10.0,1000)
res = stats.norm(0,4).pdf(xres)
#newy = np.convolve(y,res)#,'same')
newy = sig.convolve(y,res,'same')

print(len(newy),len(y))


plt.figure()
plt.subplot(1,2,1)
plt.plot(x,y)

#newx = (600/len(newy))*np.linspace(0,len(newy)-1,len(newy))
#print(newx)
plt.plot(x,newy/sum(res))


plt.subplot(1,2,2)
plt.plot(xres,res)

plt.show()
