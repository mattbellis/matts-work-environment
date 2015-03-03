#!/usr/bin/env python

from ROOT import *
import sys

filename = sys.argv[1]
val_to_plot = int(sys.argv[2])

nbins = 100

file = open(filename,"r")

h = []
h.append(TH1F("h","h",nbins,0,10.0))

count = 0
for line in file:

    if count%100000==0:
        print count

    val = line.split()[val_to_plot]
    if val=="inf":
        val = 100000.0
    elif val=='nan' or val=='-nan':
        val = -1.0
    else:
        val = float(val)

    h[0].Fill(val)

    count += 1


gStyle.SetOptStat(111111)
h[0].SetMinimum(0)
h[0].SetFillColor(22)
h[0].Draw()
gPad.Update()

print h[0].Integral(21,100)/h[0].Integral()



##########################################################
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

