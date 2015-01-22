import numpy as np
import matplotlib.pylab as plt

x = [88.0, 87.0, 86.0, 84.0]

y = [3.170000e-10,  3.171442e-10,  3.172825e-10,  3.179446e-10]

plt.figure()
plt.plot(x,y,'ko')
plt.xlim(82,90)

beta = 0.2
xd = (np.array(x[1:])/x[0])**beta
yd = np.array(y[1:])/y[0]

plt.figure()
plt.plot(xd,yd,'ko',markersize=5)
plt.ylim(0.95,1.05)

plt.show()


