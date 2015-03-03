#!/usr/bin/env python

# Import the needed modules
import sys
import os

import array

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText
from ROOT import gROOT, gStyle, gPad, TLegend, TLine, TRandom3

batchMode = False

lifetime = float(sys.argv[1])
max = int(sys.argv[2])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(0);

h = []
h.append(TH1F("h0","Data",100,-20,20));
h.append(TH1F("h1","Data",100,-20,20));
h.append(TH1F("h2","Data",100,-20,20));

rnd = TRandom3()

num = [0.0, 0.0]
for i in range(0,max):
  for j in range(0,2):
    num[j] = rnd.Exp(lifetime)
    h[j].Fill(num[j])

  h[2].Fill( num[0] - num[1] )
  
can = TCanvas("can","Data",10,10,1400,500)
can.SetFillColor(0)
can.Divide(3,1)


for j in range(0,3):
  can.cd(j+1)
  h[j].SetFillColor(43)
  if j==0:
    h[j].GetXaxis().SetTitle("t_{0}")
  elif j==1:
    h[j].GetXaxis().SetTitle("t_{1}")
  elif j==2:
    h[j].GetXaxis().SetTitle("#Delta t")
  h[j].Draw()
  gPad.Update()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



