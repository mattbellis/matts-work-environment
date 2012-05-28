#!/usr/bin/env python

import sys

import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *


################################################
def main():

  infile = open("num_count.txt", "r")

  xlo = [ 1350,  1350, 1400, 1400, 1450, 1600, 1700]
  h = []
  for i in range(0,7):
    name = "h%d" % (i)
    h.append(TH1F(name, name, 50, xlo[i], xlo[i]+300+(300*(i/6))))

  for line in infile:
    vals = line.split()
    for i in range(0,7):
      h[i].Fill(float(vals[i]))

  can = TCanvas("can", "can", 10, 10, 900, 900)
  can.SetFillColor(0)
  can.Divide(3,3)

  for i in range(0,7):
    can.cd(i+1)
    h[i].Draw()
    gPad.Update()

  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]



################################################
if __name__ == "__main__":
  main()
