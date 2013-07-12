#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import *

from color_palette import *

batchMode = False

#
# Parse the command line options
#
infilename = sys.argv[1]

gROOT.Reset()
#gStyle.SetOptStat(0)
gStyle.SetOptStat(10)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.23)
gStyle.SetPadColor(0)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)

f = TFile(infilename)

#c1.Draw("col")
map = c1.FindObject("map")

can = TCanvas("can","can",10,10,1200,800)
can.SetFillColor(0)
can.Divide(1,1)

can.cd(1)
print map.GetEntries()
map.SetMaximum(100000)
#gPad.SetLogz()
#map.Draw("colz")
map.Draw("lego2")
gPad.Update()

nx = map.GetNbinsX()
ny = map.GetNbinsY()

print "nx: %d\nny: %d" % (nx,ny)

tot = 0.0
'''
for i in range(1,nx+1):
    for j in range(1,ny+1):
        tot += map.GetBinContent(i,j)
'''

print "tot: %f" % (tot)




################################################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
