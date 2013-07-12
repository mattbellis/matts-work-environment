import numpy as np
import sys

infile = open(sys.argv[1])

for line in infile:
    sequence = np.array([list(line)])
    sequence[sequence=='T'] = 'U'
    print np.str.join('',sequence[0])
    
