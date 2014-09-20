#!/usr/bin/env python

import sys
from math import *

filename0 = sys.argv[1]
filename1 = sys.argv[2]

files = []
files.append(open(filename0,"r"))
files.append(open(filename1,"r"))

nfail_status = [[0,0], [0,0]]
npass_status = [[0,0], [0,0]]
sigs = [[],[]]

for i,file in enumerate(files):
    
    for line in file:


        ############################################################################
        # Count pass/fail status
        vals = line.split()
        status0 = vals[2]
        status1 = vals[10]
        if status0=="3":
            npass_status[i][0] += 1
        else:
            nfail_status[i][0] += 1

        if status1=="3":
            npass_status[i][1] += 1
        else:
            nfail_status[i][1] += 1

        sig0 = float(vals[3])
        sig1 = float(vals[11])
        sig = vals[1]
        goodFit = True
        #if sig=='nan' and abs(sig0-sig1)<1.00 and status1=="3":
        if sig=='nan' and status1=="3":
            sig = 0.0
            #print line
        elif sig=='nan' and status1!="3":
            #sig = 0.0
            sig = -1.0
            goodFit=False
            print line
        elif sig=='inf':
            sig = 999.9
            print "INF!"
            goodFit=False
            print line
            #sigs
        else:
            sig = float(sig)

        if status1=="3" and goodFit:
            sigs[i].append(sig)
        else:
            #print line.strip()
            1

        


# Get the 99pct from the first file
sigs[0].sort()
sigs[1].sort()
nentries = len(sigs[0])
cutoff99 = sigs[0][int(0.95*nentries)]
cutoff50 = sigs[0][int(0.50*nentries)]
#print nentries
#print int(0.95*nentries)

#print sigs[1]
#print sigs[0]

# Figure out how much is left in the other file
pct99 = 0.0
nentries = len(sigs[1])
for i in range(0,nentries):
    if sigs[1][i]>cutoff99:
        pct99 = 1.0 - i/float(nentries)
        break

print "sig 99pct: %f %f\t\tnums: %d %d" % (cutoff99, pct99, len(sigs[0]), len(sigs[1]))
#print "2.0/2.6/2.7/2.8/2.9/3.0/3.2/3.4 value: %f %f %f %f %f %f %f %f" % (pct20, pct26, pct27, pct28, pct29, pct30, pct32, pct34)
for i in range(0,2):
    for p,f in zip(npass_status[i],nfail_status[i]):
        tot = p+f
        print "%4d %4d %4d  %f" % (tot,p,f,f/float(tot))



##########################################################

