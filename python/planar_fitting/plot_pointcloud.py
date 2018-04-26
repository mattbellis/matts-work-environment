import numpy as np
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sys

infile = sys.argv[1]

X,Y,Z = np.loadtxt(infile,delimiter=',',unpack=True,dtype=float)

fig = plt.figure()
ax = fig.gca(projection='3d')


# Plot the surface.
ax.plot(X, Y, Z, '.',markersize=1)

# Customize the z axis.
#ax.set_zlim(0, 2)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

