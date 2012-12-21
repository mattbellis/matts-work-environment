#!/usr/bin/env python

import sys

infile_name = sys.argv[1]
outfile_name = 'test.tex'

infile = open(infile_name)
outfile = open(outfile_name,'w+')

for line in infile:
    print line
    if line.find('bibitem') != -1:
        start = line.find('{') + 1
        end =   line.find('}')
        reference = line[start:end]
        print reference
        output = "%s[%s]{%s" % (line.split('{')[0],reference,line.split('{')[1])
        print output
        outfile.write(output)
    else:
        outfile.write(line)


outfile.close()
