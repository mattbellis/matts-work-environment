#!/usr/bin/env python

import sys
import json

for filename in sys.argv[1:-1]:
    print filename
    f = open(filename)
    x = json.load(f)

    #jets = x["Collections"]["Jets_V1"]
    jets = x["Collections"]["BasicClusters_V1"]
    total = 0
    for j in jets:
        total += j[0]
        print total
