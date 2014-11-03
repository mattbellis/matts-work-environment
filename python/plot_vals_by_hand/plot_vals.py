import matplotlib.pylab as plt
import numpy as np
import sys

# Holders for your x and y points. Note that these are just
# python 'lists', which are different than numpy 'array' objects.
# The array object can do more, but lists are sometimes
# easier to work with.
xpts = [128,256,512,1024,2048,4096,8192,16384]
ypts = [0.003,0.046,0.120,0.391, 0.882,3.399,17.268,108.816]


# Plot and format!
plt.plot(xpts,ypts,'bo',markersize=8)
plt.xlabel('N (as in NxN symmetric matrix)')
plt.ylabel('seconds')
plt.xscale('log')
plt.yscale('log')
#plt.xlim(0,10)

plt.show()
