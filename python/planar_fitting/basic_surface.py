import numpy as np
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X, Y)
Z = 0.5*np.exp((-(X-6.0)**2)/14.5)*np.exp((-(Y-1.0)**2)/1.5)
#Z += -0.2*np.exp((-(X+6.0)**2)/17.5)*np.exp((-(Y+1.0)**2)/1.5)
a = 4
Z -= 0.1*np.cosh(X/a)*(0.5*np.cosh(Y/a))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


