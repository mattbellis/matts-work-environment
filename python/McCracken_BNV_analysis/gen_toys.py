import numpy as np
import sys

ranges = [[1.07,1.20]]
width = ranges[0][1]-ranges[0][0]

nsamples = int(sys.argv[1])
nevents = 142

for n in range(nsamples):
    name = "toys/ks_nu_toy%04d.dat" % (n)
    outfile = open(name,'w')
    events = width*np.random.random(nevents) + ranges[0][0]

    output = ""
    for ev in events:
        output += "%f\n" % (ev)
    outfile.write(output)
    outfile.close()


