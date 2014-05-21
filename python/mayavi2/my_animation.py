from enthought.mayavi import mlab
from numpy import *
import numpy as numpy

# Figure on which to draw stuff
mlab.figure(1, size=(600,600), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

l = mlab.plot3d(x, y, z, numpy.sin(mu), tube_radius=0.025, colormap='Spectral')



s = (0.0,0.8)

x0 = 0.0
y0 = 0.0
z0 = 0.0

'''
for i in range(0,10):

    theta = numpy.arccos(0.01*i)
    phi = 0.0
    r = 0.7

    x1 = r*numpy.sin(theta)*numpy.cos(phi)
    y1 = r*numpy.sin(theta)*numpy.sin(phi)
    z1 = r*numpy.cos(theta)

    x = (x0,x1)
    y = (y0,y1)
    z = (z0,z1)

    # View it.
    l = mlab.plot3d(x, y, z, s, tube_radius=0.025, colormap='Spectral')

mlab.show()
'''


#'''
# Now animate the data.
ms = l.mlab_source
for i in range(100):
  x = numpy.cos(mu)*(1+numpy.cos(n_long*mu/n_mer + numpy.pi*(i+1)/5.)*0.5)
  scalars = numpy.sin(mu + numpy.pi*(i+1)/5)
  ms.set(x=x, scalars=scalars)
#'''
