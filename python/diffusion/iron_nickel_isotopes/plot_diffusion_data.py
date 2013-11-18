import numpy as np
import matplotlib.pylab as plt

import sys

infile = open(sys.argv[1])

x = []
y = []

for i,line in enumerate(infile):

    if i!=0 and i<181:

        vals = line.split(',')

        y.append(float(vals[1]))
        x.append(float(vals[5]))


print x

plt.plot(x,y)
#plt.ylim(-1.0,5.0)
plt.show()
