from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import sys

minlh = 14864.9652913

infile = open(sys.argv[1])

xsec,mass,lh = np.loadtxt(sys.argv[1],usecols=(0,1,2),unpack=True)

#print xsec

#x = arange(1,9,40)
#y = arange(1,6,6)
y = np.arange(1e-42,1e-41,1e-42)
y = np.append(y,np.arange(1e-41,1e-40,1e-41))
y = np.append(y,np.arange(1e-40,1e-39,1e-40))
y = np.append(y,np.arange(1e-39,1e-38,1e-39))
print y
#x = np.log10(x)
x = np.array([6,8,10,15,20,30])
X,Y = meshgrid(x, y) # grid of point
print X
print Y


Z = np.ones_like(X)
Z *= 0

#print Z

for i,j,k in zip(xsec,mass,lh):
    ii = jj = None
    for index,y in enumerate(Y[:,0]):
        if i==y:
            ii = index
            #print i,j,k,ii
            break
    for index,x in enumerate(X[0]):
        if j==x:
            jj = index
            #print i,j,k,jj
            break

    #print np.exp(minlh-k)
    #print ii,jj,i,j,k,np.exp(minlh-k)
    if ii is not None and jj is not None:
        #print jj,ii
        print np.exp(minlh-k)
        if np.exp(minlh-k) > 0.0:
            Z[ii][jj] = np.log10(np.exp(minlh-k))
        #Z[ii][jj] = j
        #Z[ii][jj] = k
        '''
        if ii==5 and jj==5:
            #Z[ii][jj] = 1.0
            print k
            print "JDFKJDSKJF"
        '''

    



'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                              cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#ax.set_xscale('log')

fig.colorbar(surf, shrink=0.5, aspect=5)
'''

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# Contours
im = imshow(Z,cmap=cm.RdBu,extent=(min(mass),max(mass),min(xsec),max(xsec)),origin='lower') # drawing the function
# adding the Contour lines with labels
#cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
#clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
#print X
#print Y
ax.set_yscale('log')
# latex fashion title

plt.show() 
