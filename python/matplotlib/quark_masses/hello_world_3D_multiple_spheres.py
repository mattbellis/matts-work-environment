from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

masses = [0.002, 0.006, 0.100, 3.0, 5.0, 170.0]
xpos   = [0.66, -0.33, 0.66, -0.33, 0.66, -0.33]
ypos   = [0, 0, 5, 5, 10, 10]
color   = ['g','g','b','b','r','r']

for j,m in enumerate(masses):

    i = 5-j

    r = pow(masses[i],0.33)/10.0
    print r/10.0
    #r = 0.1

    zpos = r

    x = xpos[i] + r * np.outer(np.cos(u), np.sin(v))
    y = ypos[i] + r * np.outer(np.sin(u), np.sin(v))
    z = zpos    + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color[i])



ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,14)
ax.set_zlim3d(0,1)

plt.show()
