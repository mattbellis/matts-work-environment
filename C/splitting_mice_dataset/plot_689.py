import numpy as np
import matplotlib.pylab as plt
import sys

infile = open(sys.argv[1])

x,y,z = np.loadtxt(infile, delimiter=' ',unpack=True,usecols=(4,5,6))

plt.figure(figsize=(13,4))
plt.subplot(1,3,1)
plt.plot(x,y,'o',markersize=1)

plt.subplot(1,3,2)
plt.plot(x,z,'o',markersize=1)

plt.subplot(1,3,3)
plt.plot(y,z,'o',markersize=1)


plt.show()
