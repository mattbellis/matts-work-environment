from enthought.mayavi import mlab
from numpy import *
import numpy as numpy

# Produce some nice data.
#n_mer, n_long = 6, 11
#pi = numpy.pi
#dphi = pi/1000.0
#phi = numpy.arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')
#mu = phi*n_mer
'''
x = numpy.cos(mu)*(1+numpy.cos(n_long*mu/n_mer)*0.5)
y = numpy.sin(mu)*(1+numpy.cos(n_long*mu/n_mer)*0.5)
z = numpy.sin(n_long*mu/n_mer)*0.5
'''


s = (0.0,0.8)

for i in range(0,100):
    theta = numpy.acos(0.1*i)
    phi = 0.0
    r = 0.7
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0

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
# Now animate the data.
ms = l.mlab_source
for i in range(100):
  x = numpy.cos(mu)*(1+numpy.cos(n_long*mu/n_mer + numpy.pi*(i+1)/5.)*0.5)
  scalars = numpy.sin(mu + numpy.pi*(i+1)/5)
  ms.set(x=x, scalars=scalars)
'''
