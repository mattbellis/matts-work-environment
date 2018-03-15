import numpy as np
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

X = []
Y = []
stepsize = 0.02
# Make data.
npts = 50
for i in range(0,npts):
    for j in range(0,npts):
        X.append(i*stepsize)
        Y.append(j*stepsize)

X = np.array(X)
Y = np.array(Y)

# X line
mx = 1; bx = 1
my = 1; by = 0
Z = np.ones(len(X))
for i,(x,y) in enumerate(zip(X,Y)):
    #Z[i] = (mx*x + bx) + ( my*y + by)
    Z[i] = np.exp((-(x-0.5)**2)/0.02)*np.exp((-(y-1.5)**2)/0.52)





# Plot the surface.
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot(X, Y, Z, '.',markersize=1)

# Customize the z axis.
ax.set_zlim(0, 2)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

