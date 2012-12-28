import numpy as np
import sys

infile = open(sys.argv[1])

bases = ['G','A','C','T']
for line in infile:
    sequence = np.array([list(line)])
    for b in bases:
        print "# of %s: %d" % (b,len(sequence[sequence==b]))
    
