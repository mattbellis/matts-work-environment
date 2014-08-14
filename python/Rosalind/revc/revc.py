import numpy as np
import sys

infile = open(sys.argv[1])

bases = ['G','A','C','T']
complement = ['C','T','G','A']
for line in infile:
    sequence = np.array([list(line)])
    for i,b in enumerate(bases):
        sequence[sequence==b] = str(i)
    for i,b in enumerate(complement):
        sequence[sequence==str(i)] = b

    reversed_sequence = np.array([sequence[0][::-1]])
    print np.str.join('',reversed_sequence[0])
    
