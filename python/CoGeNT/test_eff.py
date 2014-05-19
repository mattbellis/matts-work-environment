import numpy as np
import scipy as sp

import scipy.integrate as integrate

import matplotlib.pylab as plt

slope = -3.36

x = np.linspace(0.5,3.2,1000)
y = np.exp(slope*x)

norm = integrate.simps(y,x=x)
print norm

y /= norm
y *= 575.0*0.025

plt.plot(x,y)
plt.show()
