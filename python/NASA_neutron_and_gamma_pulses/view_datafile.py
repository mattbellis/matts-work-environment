import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
from scipy.stats import lognorm

import sys

infilename = sys.argv[1]
vals = np.loadtxt(infilename,skiprows=5,delimiter=',',unpack=True,dtype=float)

t = vals[0]
v = vals[1]

print len(t)

plt.plot(t,v,'bo',markersize=1)


plt.show()


