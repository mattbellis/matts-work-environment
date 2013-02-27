import numpy as np
import matplotlib.pylab as plt

################################################################################
# Generate random numbers from a flat distribution
# From 0-100
################################################################################
x = 100*np.random.random(1000000)

plt.figure()
plt.hist(x,bins=20)



################################################################################
# Generate random numbers from a Gaussian distribution
################################################################################
mean = 10.0
sigma = 3.0
x = np.random.normal(mean,sigma,1000000)

plt.figure()
plt.hist(x,bins=200)

################################################################################
# Generate random numbers from a sin^2 distribution
# We will use an accept-and-reject method.
################################################################################

max_val = 1.0
x_range = 10.0
npts = 100000

period = 2*np.pi

i = 0
x = []
while i<npts:

   xpt = 10*np.random.random()

   val = (np.sin(xpt))**2

   test = max_val*np.random.random()

   if test<val:
       x.append(xpt)
       i += 1

plt.figure()
plt.hist(x,bins=200)



plt.show()

