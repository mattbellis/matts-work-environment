import sys
import numpy as np

lines = []

infilename = sys.argv[1]
infile = open(infilename)

for line in infile:
    lines.append(line)

print lines[0]
np.random.shuffle(lines)
print lines[0]

outfilename = "%s_randomized.%s" % (infilename.split('.')[0],infilename.split('.')[1])
print outfilename
outfile = open(outfilename,'w')
for line in lines:
    outfile.write(line)
outfile.close()
