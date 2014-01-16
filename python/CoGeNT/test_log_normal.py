from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import lichen.pdfs as pdfs

X = np.linspace(0,5.0,50)
Y = np.linspace(0.5,3,50)
X, Y = np.meshgrid(X, Y)
print X
print Y
#R = np.sqrt(X**2 + Y**2)
# Fast
mu = [-0.384584,-0.473217,0.070561]
sigma = [0.751184,-0.301781,0.047121]
#amp = [0.126040,0.220238,-0.032878]
#mu = [0.5,0.0,0.0]
#sigma = [0.5,0.0,0.0]
#amp = [1.0,-0.2,0.1]
Zf = pdfs.lognormal2D_unnormalized(X,Y,mu,sigma)

for i in range(0,len(Zf)):
    val = max(Zf[i])
    print "maxval: ",val
    Zf[i] /= float(val)

#print Zf

# Slow
mu = [0.897153,-0.304876,0.044522]
sigma = [0.126040,0.220238,-0.032878]
#amp = [0.873944,-0.220215,0.032871]
Zs = pdfs.lognormal2D_unnormalized(X,Y,mu,sigma)

for i in range(0,len(Zs)):
    val = max(Zs[i])
    print "maxval: ",val
    Zs[i] /= float(val)

#ax.set_zlim(0.00, 1.01)

print len(X)
#print len(X[0])
print len(Y)
#print len(Y[0])
print len(Zf)
figf = plt.figure()
axf = figf.gca(projection='3d')
surff = axf.plot_surface(X, Y, Zf, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surff = axf.contour(X, Y, Zf, linewidth=0, antialiased=False)
axf.zaxis.set_major_locator(LinearLocator(10))
axf.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
figf.colorbar(surff, shrink=0.5, aspect=5)

figs = plt.figure()
axs = figs.gca(projection='3d')
surfs = axs.plot_surface(X, Y, Zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs.zaxis.set_major_locator(LinearLocator(10))
axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
figs.colorbar(surfs, shrink=0.5, aspect=5)

plt.show()

