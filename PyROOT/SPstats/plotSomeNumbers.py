#!/usr/bin/env python

# Import the needed modules
import sys
import os

import array

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText
from ROOT import gROOT, gStyle, gPad, TLegend, TLine

#def plotAll(viewmode, fileformat="pdf", batchMode=False):

batchMode = False

filename = sys.argv[1]
modenum = sys.argv[2]

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(0);

h = TH1F("h","Data",200,80,120);
#Open the input file 
if os.path.isfile(filename):
  file = open(filename,"r")

  for line in file:
    entries = line.split()
    val = float(entries[0])
    h.Fill(float(val))

  can = None
  for i in range(0,1):
    name = "can" + str(i)
    title = "Mode " + modenum
    can = TCanvas(name, title, 10+10*i, 10+10*i, 1200, 1000);
    can.SetFillColor(0)
    can.Divide(1,1)

  can.cd(1)

  h.GetYaxis().SetTitle("Generated events")
  h.GetYaxis().SetTitle("% req events")
  h.SetTitle("")

  h.GetYaxis().SetTitleSize(0.09)
  h.GetYaxis().SetTitleFont(42)
  h.GetYaxis().SetTitleOffset(0.7)
  h.GetYaxis().CenterTitle()
  h.GetYaxis().SetNdivisions(6)

  h.GetXaxis().SetNdivisions(4)
  h.GetXaxis().SetNoExponent(False)

  color = 36
  h.SetLineWidth(2)
  h.SetLineColor(color)
  h.SetFillColor(color)
  h.SetMarkerColor(color)

  can.cd(1)
  gPad.SetLeftMargin(0.30);
  gPad.SetTopMargin(0.12);
  gPad.SetBottomMargin(0.15);

  text = TPaveText(0.00, 0.95, 0.45, 0.99,"NDC")
  name = "mode: %d" % int(modenum)
  text.AddText(name);
  text.SetFillColor(1)
  text.SetTextColor(0)
  text.Draw()
        

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
#if (1):
  print "a"
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



