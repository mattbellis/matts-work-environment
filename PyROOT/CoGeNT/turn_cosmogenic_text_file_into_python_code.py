#!/usr/bin/env python

import sys
import csv

filename = sys.argv[1]

infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')

print "cosmogenic_data_dict = {"
for i,line in enumerate(infile):

    output = ""

    if i>0:

        output += "\"%s\": [" % (line[0])
        
        nvals = len(line)
        for j in range(1,nvals):

            output += "%s" % (line[j])

            if j<nvals-1:
                output += ","
            else:
                output += "], "

        print output

print "}"
