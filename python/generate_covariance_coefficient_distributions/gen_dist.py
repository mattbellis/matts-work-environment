import matplotlib.pylab as plt
import numpy as np

mean = [0,0]
cov = [[1,-0.9],[-0.9,1]] # diagonal covariance, points lie on x or y-axis

x,y = np.random.multivariate_normal(mean,cov,1000).T

plt.figure()
plt.plot(x,y,'o')
plt.axis('equal')

plt.figure()
plt.hist(x,bins=25)

plt.figure()
plt.hist(y,bins=25)

print x
print y

zipped = zip(x,y)
np.savetxt('dataset4.dat',zipped,fmt='%f %f')

plt.show()
