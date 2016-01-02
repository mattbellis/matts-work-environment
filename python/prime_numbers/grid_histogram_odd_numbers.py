import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from is_prime import is_prime,is_square,is_odd

import lichen.lichen as lch

import sys

# Gen list of primes

numbers = np.arange(3,100000,1)

odds = []
for n in numbers:
    if is_odd(n):
        odds.append(n)

#print odds

nodds = len(odds)

print nodds 

#nodds = int(sys.argv[1])
#maxodd = nodds
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
for i in range(0,maxodd-1):
    #for j in range(i+1,maxodd):
    for j in range(i+1,i+2):
        a = odds[i]
        b = odds[j]

        c = (a*b) - (b+a)

        val = is_prime(c)

        if val:
            nwork += 1
            #grid[i][j] = 1
            #grid[j][i] = 1
            xpts.append(i)
            ypts.append(j)

            xpts.append(j)
            ypts.append(i)
            print a,b,c
        else:
            val = is_square(c)
            if val:
                xsquare.append(i)
                ysquare.append(j)

                xsquare.append(j)
                ysquare.append(i)
            else:
                print "NOT! %d %d %d" % (a,b,c)

        #print "%4d %4d %7d %s" % (a,b,c,val)
        ntot += 1

print "---------"
#print odds
print (len(odds))
print maxodd
print nodds
print ntot
print nwork
print ntot-nwork
print float(nwork)/ntot

plt.figure()
#plt.imshow(grid,origin='upper',cmap = cm.Greys_r)
#plt.plot(xsquare,ysquare,'ro',markersize=1)
#plt.ylim(maxodd,0)
#plt.xlim(0,maxodd)

lch.hist_2D(xpts,ypts,xbins=50,ybins=50)

name = 'numbers_grid_histo_%d.png' % (nodds)
plt.savefig(name)

plt.tight_layout()

plt.show()
