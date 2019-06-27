import numpy as np

import sys

infilename = sys.argv[1]
outfilename = '{0}_DECODED.csv'.format(infilename.split('.txt')[0])

infile = open(infilename,encoding='utf8', errors='ignore')
outfile = open(outfilename,'w')

for line in infile:

    vals = line.replace('\x00','').split('\t')

    print(vals)

    if len(vals)==1 and vals[0]=='\n':
        continue

    output = ','.join(vals)
    output.replace('\n','')
    output.rstrip('\n')
    print(output)
    outfile.write(output)

outfile.close()




