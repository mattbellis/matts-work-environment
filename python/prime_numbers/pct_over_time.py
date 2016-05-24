import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from is_prime import is_prime,is_square

import sys

# Gen list of primes

numbers = np.arange(3,50000,1)

primes = []
for n in numbers:
    if is_prime(n):
    #if n%2==1:
        primes.append(n)

#print primes

nprimes = len(primes)

print nprimes 

#nprimes = int(sys.argv[1])
#maxprime = nprimes
maxprime = int(sys.argv[1])
nprimes = maxprime


xsquare = []
ysquare = []

block_size = 10
nblocks = int(maxprime/block_size)

ntot = np.zeros((nblocks,nblocks))
nwork = np.zeros((nblocks,nblocks))

biggest_prime = 3

for ik in range(0,nblocks):

    istart = ik*block_size
    imax = (ik+1)*block_size

    print ik,nblocks

    for i in range(istart,imax):

        for jk in range(0,nblocks):

            jstart = jk*block_size
            jmax = (jk+1)*block_size

            for j in range(jstart,jmax):

                a = primes[i]
                b = primes[j]

                #print ik,jk,a,b

                c = (a*b) - abs((b-a))

                val = is_prime(c)

                if val:
                    nwork[ik][jk] += 1
                    if c>biggest_prime:
                        biggest_prime=c
                        print a,b,c

                else:
                    val = is_square(c)
                    if val:
                        xsquare.append(i)
                        ysquare.append(j)

                        xsquare.append(j)
                        ysquare.append(i)

                ntot[ik][jk] += 1
            #print "%4d %4d %7d %s" % (a,b,c,val)

#print nwork[5][6]
#print nwork[6][5]

plt.figure()
plt.imshow(ntot,origin='upper',cmap = cm.coolwarm)
plt.ylim(nblocks,0)
plt.xlim(0,nblocks)
plt.tight_layout()
name = 'numbers_tot_%d.png' % (nprimes)
plt.savefig(name)

plt.figure()
plt.imshow(nwork,origin='upper',cmap = cm.coolwarm)
plt.ylim(nblocks,0)
plt.xlim(0,nblocks)
plt.tight_layout()
name = 'numbers_work_%d.png' % (nprimes)
plt.savefig(name)

yvals_work = np.zeros(nblocks)
yvals_tot = np.zeros(nblocks)
x = block_size*np.arange(0,nblocks)
for i in range(0,nblocks):
    w = nwork[0:i,0:i]
    t = ntot[0:i,0:i]
    if len(w)<1:
        print w
        print sum(w)
        yvals_work[i] = sum(w)
        yvals_tot[i] = sum(t)
    elif len(w)>=1:
        yvals_work[i] = sum(sum(w))
        yvals_tot[i] = sum(sum(t))

print x
print yvals_tot
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.plot(x,yvals_tot,'bo')

plt.subplot(1,3,2)
plt.plot(x,yvals_work,'bo')

plt.subplot(1,3,3)
plt.plot(x,yvals_work/yvals_tot,'bo')
plt.xlim(-block_size,(nblocks+1)*block_size)
plt.ylim(0)

plt.tight_layout()

name = 'numbers_1D_%d.png' % (nprimes)
plt.savefig(name)
plt.show()
