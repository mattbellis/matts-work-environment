from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import lichen.pdfs as pdfs

X = np.linspace(0,5.0,25)
Y = np.linspace(0.5,3,25)
X, Y = np.meshgrid(X, Y)
print X
print Y
#R = np.sqrt(X**2 + Y**2)
# Fast
mu = [0.0705606,-0.4732166,-0.38458368]
sigma = [0.0318007,-0.12876697,0.6538349]
#amp = [-0.03287787,0.22023833,0.12603986]
#mu = [0.5,0.0,0.0]
#sigma = [0.5,0.0,0.0]
amp = [1.0,-0.2,0.1]
Zf = pdfs.lognormal2D_unnormalized(X,Y,mu,sigma)
maxval = 0.0
'''
for i in range(0,len(Zf)):
    if val>maxval:
        maxval = val
'''

print "maxval: ",maxval
val = 1.0
for i in range(0,len(Zf)):
    val = max(Zf[i])
    print "maxval: ",val
    for j in range(0,len(Zf[i])):
        Zf[i][j] /= val

print Zf

# Slow
mu = [0.04452242,-0.30487632,0.89715279]
sigma = [0.04712092,-0.30178065,0.7511839]
#amp = [0.03287055,-0.22021459,0.87394406]
Zs = pdfs.lognormal2D_unnormalized(X,Y,mu,sigma)

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

