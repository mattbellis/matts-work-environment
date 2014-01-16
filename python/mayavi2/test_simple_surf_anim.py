from enthought.mayavi import mlab
from numpy import *
import numpy as numpy

x, y = numpy.mgrid[0:3:1,0:3:1]
s = mlab.surf(x, y, numpy.asarray(x*0.1, 'd'))

for i in range(1):
    s.mlab_source.scalars = numpy.asarray(x*0.1*(i+1), 'd')

