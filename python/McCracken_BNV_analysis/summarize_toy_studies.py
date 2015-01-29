import matplotlib.pylab as plt
import numpy as np

import sys

infile = open(sys.argv[1])

junk,nsig,err = np.loadtxt(infile,dtype=str,usecols=(0,1,2),unpack=True)

nsig = nsig.astype('float')
err = err.astype('float')

#print nsig
#print err

print np.mean(nsig)
significance = nsig/err
ntoys = len(significance)
ngt3 = len(significance[significance>3.0])

print "%d %d %f" % (ntoys,ngt3,float(ngt3)/ntoys)

plt.figure()
plt.hist(nsig,bins=25)

plt.figure()
plt.plot(nsig,err,'bo',markersize=2)

#plt.show()
