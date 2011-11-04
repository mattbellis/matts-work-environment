#!/usr/bin/env python

import sys
import json

files = sys.argv[1:-1]
if len(sys.argv)==2:
    files = [sys.argv[1]]
for filename in files:
    print filename
    f = open(filename)
    x = json.load(f)

    # Dump the Types
    print x.keys()
    mykey = "Types"
    mykey = "Associations"
    types = x[mykey]
    for t in types:
        print t
        vals = x[mykey][t]
        output = "%s\n" % (t)
        for val in vals:
            for v in val:
                output += "\t%s"  % (v)
            output += "\n"
        print output

    #jets = x["Collections"]["Jets_V1"]
    jets = x["Collections"]["BasicClusters_V1"]
    total = 0
    for j in jets:
        total += j[0]
        print total
