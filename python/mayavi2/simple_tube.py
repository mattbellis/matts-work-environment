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


x = (0.0, 0.5)
y = (0.0, 0.5)
z = (0.0, 0.5)

s = (0.0,0.8)

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
