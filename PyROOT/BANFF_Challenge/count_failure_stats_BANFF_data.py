#!/usr/bin/env python

import sys

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
        if vals[1]=="3":
            npass_status[i][0] += 1
        else:
            nfail_status[i][0] += 1

        if vals[9]=="3":
            npass_status[i][1] += 1
        else:
            nfail_status[i][1] += 1

        sig = vals[0]
        if sig=='nan':
            sigs[i].append(0.0)
            #sigs
        else:
            sigs[i].append(float(sig))

        


# Get the 99pct from the first file
sigs[0].sort()
sigs[1].sort()
nentries = len(sigs[0])
cutoff99 = sigs[0][int(0.99*nentries)]
cutoff50 = sigs[0][int(0.50*nentries)]

# Figure out how much is left in the other file
nentries = len(sigs[1])
for i in range(0,nentries):
    if sigs[1][i]>cutoff99:
        pct99 = 1.0 - i/float(nentries)
        break

print "sig 99pct: %f %f" % (cutoff99, pct99)
#print "2.0/2.6/2.7/2.8/2.9/3.0/3.2/3.4 value: %f %f %f %f %f %f %f %f" % (pct20, pct26, pct27, pct28, pct29, pct30, pct32, pct34)
for i in range(0,2):
    for p,f in zip(npass_status[i],nfail_status[i]):
        tot = p+f
        print "%4d %4d %4d  %f" % (tot,p,f,f/float(tot))



##########################################################

