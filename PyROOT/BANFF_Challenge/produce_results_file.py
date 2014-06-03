#!/usr/bin/env python

import sys
from math import *

sig99 = float(sys.argv[1])
filename = sys.argv[2]

infile = open(filename,'r')

for line in infile:
    vals = line.split()
    dataset = int(vals[0])
    sig = vals[1]

    if sig=='nan':
        sig = 0.0
    else:
        sig = float(vals[1])

    nvals = []
    for i in range(0,6):
        nvals.append(float(vals[4+i]))

    # Calc error on final number
    pcterr1 = nvals[1]/nvals[0]
    pcterr2 = nvals[3]/nvals[2]
    nevents = nvals[0]+nvals[2]+nvals[4]
    #nvals[5] = nvals[4]*sqrt(pcterr1*pcterr1 + pcterr2*pcterr2 - 2.0*0.9*pcterr1*pcterr2)
    nvals[5] = sqrt(nvals[1]*nvals[1] + nvals[3]*nvals[3] - 2.0*0.85*nvals[1]*nvals[3])


    is_signal = 'no'
    if sig>sig99:
        is_signal = 'yes'
    else:
        is_signal = 'no'

    print "%d %3s %4.3f   %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f" % (dataset,is_signal,sig,nvals[0],nvals[1],nvals[2],nvals[3],nvals[4],nvals[5])


