import numpy as np
import sys

tot = np.zeros((16,16,16)).astype('int')

for i in xrange(int(sys.argv[1])):
    name = "test_out_%05d.dat" % (i)
    infile = open(name,'r')

    x = np.loadtxt(infile)

    newx = x.reshape(16,16,16)

    tot += newx

print tot
for data_slice in tot:
    print data_slice

