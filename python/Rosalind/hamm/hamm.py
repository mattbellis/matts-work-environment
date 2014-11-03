import numpy as np
import sys

infile = open(sys.argv[1])

bases = ['G','A','C','T']
seq0 = np.array([list(infile.readline().rstrip())])
seq1 = np.array([list(infile.readline().rstrip())])

locations0 = []
locations1 = []

 
hamm_dist = len(seq0[0])
for b in bases:
    loc0 = seq0==b 
    loc1 = seq1==b 

    diff = loc0*loc1
    #print diff
    same_base = diff[diff]

    #print same_base
    #print hamm_dist
    hamm_dist -= len(same_base)

print "hamm distance: %d" % (hamm_dist)

    
