import numpy as np
import sys

for i in xrange(int(sys.argv[1])):
    x = np.random.randint(500000000,size=(16,16,16))

    name = "test_out_%05d.dat" % (i)
    outfile = open(name,'w')
    for data_slice in x:
        #print data_slice
        np.savetxt(outfile,data_slice,fmt="%-10d")


    outfile.close()
