import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from is_prime import is_prime,is_square

import lichen.lichen as lch

import sys

# Gen list of primes

numbers = np.arange(3,10000,1)

primes = []
for n in numbers:
    if is_prime(n):
        primes.append(n)

#print primes

nprimes = len(primes)

print nprimes 

#nprimes = int(sys.argv[1])
#maxprime = nprimes
maxprime = int(sys.argv[1])

ntots = []
nworks = []
maxprimes = []

#grid = np.zeros((maxprime,maxprime))
xpts = []
ypts = []

xsquare = []
ysquare = []

nwork = 0
ntot = 0
for i in range(0,maxprime-1):
    for j in range(i+1,maxprime):
        a = primes[i]
        b = primes[j]

        c = (a*b) - (b-a)

        val = is_prime(c)

        if val:
            nwork += 1
            #grid[i][j] = 1
            #grid[j][i] = 1
            xpts.append(i)
            ypts.append(j)

            xpts.append(j)
            ypts.append(i)
        else:
            val = is_square(c)
            if val:
                xsquare.append(i)
                ysquare.append(j)

                xsquare.append(j)
                ysquare.append(i)

        #print "%4d %4d %7d %s" % (a,b,c,val)
        ntot += 1

print "---------"
print primes
print (len(primes))
print maxprime
print nprimes
print ntot
print nwork
print ntot-nwork
print float(nwork)/ntot

plt.figure()
#plt.imshow(grid,origin='upper',cmap = cm.Greys_r)
#plt.plot(xsquare,ysquare,'ro',markersize=1)
#plt.ylim(maxprime,0)
#plt.xlim(0,maxprime)

lch.hist_2D(xpts,ypts,xbins=50,ybins=50)

name = 'numbers_grid_histo_%d.png' % (nprimes)
plt.savefig(name)

plt.tight_layout()

plt.show()
