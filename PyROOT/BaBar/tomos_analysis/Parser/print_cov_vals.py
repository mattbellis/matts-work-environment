#!/usr/bin/env python

import sys
from math import *

i0 = int(sys.argv[1])
i1 = int(sys.argv[2])

parms_file = open("fitlog_1.log")
covmat_file = open("fitlog_0.log")

parms = []
count = 1
for line in parms_file:
    vals = line.split()
    if count==i0 or count==i1:
        parms.append(line)

    count += 1



covmat_vals = []
count = 1
for line in covmat_file:
    vals = line.split()
    if count==i0 and len(vals)>=i1:
        covmat_vals.append(float(vals[i1-1]))
    elif count==i1 and len(vals)>=i0:
        covmat_vals.append(float(vals[i0-1]))

    count += 1



for p in parms:
    print p
for c in covmat_vals:
    print "%f %f" % (c, (c/abs(c))*sqrt(abs(c)))





