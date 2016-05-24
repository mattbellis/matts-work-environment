import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from is_prime import is_prime,is_square,is_odd

import lichen.lichen as lch

import sys

# Gen list of primes

maxodd = int(sys.argv[1])

ntots = []
nworks = []
maxodds = []

#grid = np.zeros((maxodd,maxodd))
xpts = []
ypts = []

xsquare = []
ysquare = []

nwork = 0
ntot = 0
for i in range(0,maxodd):

    if i%100000==0:
        print i,maxodd

    c = 6*i + 5

    val = is_prime(c)

    if val:
        nwork += 1
        #grid[i][j] = 1
        #grid[j][i] = 1
        xpts.append(i)
        ypts.append(i)

        xpts.append(i)
        ypts.append(i)
        #print i,c
    else:
        1
        #print "NOT! %d %d %d" % (i,i*6,c)

    #print "%4d %4d %7d %s" % (a,b,c,val)
    ntot += 1

print "---------"
print maxodd
print ntot
print nwork
print ntot-nwork
print float(nwork)/ntot

'''
plt.figure()

lch.hist_2D(xpts,ypts,xbins=50,ybins=50)

name = 'numbers_grid_histo_fives_and_sixes_%d.png' % (maxodd)
plt.savefig(name)

plt.tight_layout()

plt.show()
'''
