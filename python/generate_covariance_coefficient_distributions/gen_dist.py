import matplotlib.pylab as plt
import numpy as np

mean = [0,0]
cov = [[1,0.2],[0.2,1]] # diagonal covariance, points lie on x or y-axis

x,y = np.random.multivariate_normal(mean,cov,10000).T

plt.plot(x,y,'o')
plt.axis('equal')

zipped = zip(x,y)
np.savetxt('dataset4.dat',zipped,fmt='%f %f')

plt.show()
