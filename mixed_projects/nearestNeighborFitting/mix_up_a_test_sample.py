#!/usr/bin/env python

import random

import sys

infiles = []

infiles.append(open("bc2p2bg1mc.dat","r"))
infiles.append(open("bc2p2bg2mc.dat","r"))
infiles.append(open("bc2p2sigmc.dat","r"))

outfile = open(sys.argv[4],"w+")

nevents = []

counts = []
for i in range(0,3):
    nevents.append(int(sys.argv[i+1]))

    prob = nevents[i]/5000.0

    counts.append(0)
    for line in infiles[i]:
        test = random.random()
        if test<prob:
            #print line.split()[0]
            outfile.write(line)
            counts[i] += 1

for c in counts:
    print "Nevents: %d" % (c)



