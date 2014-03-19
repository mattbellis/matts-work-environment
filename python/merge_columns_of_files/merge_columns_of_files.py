import numpy as np
import sys

outfile = open("output.csv",'w+')

infiles = sys.argv[1:]
print infiles

vals = []
for infilename in infiles:
    infile = open(infilename)
    vals.append(np.array(infile.read().split()))

#print vals
for i in range(len(vals[0])):
    output = ""
    for j in range(len(vals)):
        output += "%s," % (vals[j][i])
    output += "\n"
    #print output
    outfile.write(output)

outfile.close()
