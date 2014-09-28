import numpy as np
import sys

infile = open(sys.argv[1])

new_event = True

num = 0
beamE = 0.0
p = [[],[],[]]
for i in xrange(3):
    p[i] += [0.0,0.0,0.0]


#exit(1)

nparticle = 0
for line in infile:

    vals = line.split()

    if len(vals)==2:
        new_event = True
        num = int(vals[0])
        beamE = float(vals[1])
        nparticle = 0

    elif len(vals)==5:

        p[nparticle] = float(vals[0]),float(vals[1]),float(vals[2])

        nparticle += 1


