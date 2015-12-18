import numpy as np

from is_prime import is_prime

import sys

# Gen list of primes

numbers = np.arange(3,1000,1)

primes = []
for n in numbers:
    if is_prime(n):
        primes.append(n)

#print primes

#nprimes = len(primes)
nprimes = int(sys.argv[1])

maxprime = nprimes

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

print nprimes
print ntot
print nwork
print ntot-nwork
print float(nwork)/ntot
