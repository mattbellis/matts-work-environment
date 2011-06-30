#!/usr/bin/env python

from ROOT import *
import sys

nbins = int(sys.argv[1])
filename = sys.argv[2]

file = open(filename,"r")

h = []
h.append(TH1F("h","h",nbins,0,1.0))

count = 0
for line in file:

    if count%100000==0:
        print count

    val = float(line.split()[0])

    h[0].Fill(val)

    count += 1


h[0].SetMinimum(0)
h[0].SetFillColor(22)
h[0].Draw()



##########################################################
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

