#!/usr/bin/env python

import sys

infilename = sys.argv[1]

file_tag = 'banff'

chunk_size = 100

if len(sys.argv)>2:
    file_tag = sys.argv[2]

infile = open(infilename)

line = infile.readline()

outfile = None

subfile_count = 0
output = ""
outfilename  = ""

while not line == "":
    vals = line.split()
    sample = int(vals[0])
    nentries = int(vals[1])
    #print nentries
    output += "%d %d\n" % (sample, nentries)

    if subfile_count == 0:
        outfilename = "Problem_2_files/samples_%s_%05d-%05d.dat" % (file_tag,sample,sample+chunk_size-1)
        outfile = open(outfilename, "w")
        
    for i in range(0,nentries):
        line = infile.readline()
        val = float(line)
        #print val
        output += "%f\n" % (val)


    subfile_count += 1

    if subfile_count==chunk_size:
        subfile_count = 0
        outfile.write(output)
        output = ""
        print outfilename

    line = infile.readline()


outfile.close()


