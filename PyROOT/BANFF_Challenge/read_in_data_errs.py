#!/usr/bin/env python

from ROOT import *
import sys

nbins = int(sys.argv[1])
filename = sys.argv[2]

file = open(filename,"r")

h = []
h.append(TH1F("h0","h0",nbins,0,1.0))
h.append(TH1F("h1","h1",nbins,0,1.0))
h.append(TH1F("h2","h2",nbins,0,1.0))

h.append(TH2F("h3","h3",nbins,0,1.0,nbins,0,1.0))
h.append(TH2F("h4","h4",nbins,0,1.0,nbins,0,1.0))
h.append(TH2F("h5","h5",nbins,0,1.0,nbins,0,1.0))

count = 0
for line in file:

    if count%100000==0:
        print count

    x =   float(line.split()[0])
    y =   float(line.split()[1])
    err = float(line.split()[2])
    print "%f %f %f" % (x,y,err)

    h[0].Fill(x)
    h[1].Fill(y)
    h[2].Fill(err)

    h[3].Fill(x,y)
    h[4].Fill(x,err)
    h[5].Fill(y,err)

    count += 1


can = TCanvas("can","can",10,10,1100,700)
can.SetFillColor(0)
can.Divide(3,2)

for i in range(len(h)):
    can.cd(i+1)
    h[i].SetMinimum(0)
    h[i].SetFillColor(22)
    h[i].Draw()
    gPad.Update()



##########################################################
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

