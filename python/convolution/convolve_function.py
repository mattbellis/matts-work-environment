import matplotlib.pylab as plt
import numpy as np

import scipy.stats as stats

x = np.convolve([1, 2, 3], [0, 1, 0.5])
print(x)

x = np.linspace(0,600,1000)
y = np.exp(-x/150)

xres = np.linspace(-3,3.0,1000)
res = stats.norm(0,0.01).pdf(xres)
newy = np.convolve(y,res)#,'same')

print(len(newy),len(y))


plt.figure()
plt.subplot(1,2,1)
plt.plot(x,y)

newx = (600/len(newy))*np.linspace(0,len(newy)-1,len(newy))
print(newx)
plt.plot(newx,newy/100.0)


plt.subplot(1,2,2)
plt.plot(xres,res)

plt.show()
