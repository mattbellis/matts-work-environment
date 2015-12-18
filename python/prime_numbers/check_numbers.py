import numpy as np
import matplotlib.pylab as plt

from is_prime import is_prime

import sys

# Gen list of primes

numbers = np.arange(3,2000,1)

primes = []
for n in numbers:
    if is_prime(n):
        primes.append(n)

#print primes

nprimes = len(primes)

print nprimes 

nprimes = int(sys.argv[1])
#maxprime = int(sys.argv[1])

ntots = []
nworks = []
maxprimes = []

for maxprime in range(10,nprimes,10):

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
    

    maxprimes.append(maxprime)
    ntots.append(ntot)
    nworks.append(nwork)

nworks = np.array(nworks)
ntots = np.array(ntots)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(ntots,nworks,'bo')
plt.xlabel("# total")
plt.ylabel("# prime")

plt.subplot(1,3,2)
plt.plot(maxprimes,nworks,'bo')
plt.xlim(-10,max(maxprimes)+10)
plt.xlabel("# max primes")
plt.ylabel("# prime")

plt.subplot(1,3,3)
plt.plot(maxprimes,nworks.astype(float)/ntots,'bo')
plt.xlim(-10,max(maxprimes)+10)
plt.ylim(0)
plt.xlabel("# max primes")
plt.ylabel("fraction that are prime")

name = 'numbers_%d.png' % (nprimes)
plt.savefig(name)

plt.tight_layout()

plt.show()
