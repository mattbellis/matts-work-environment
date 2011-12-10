#!/usr/bin/env python

# Import the needed modules
import sys
import os

import array

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText
from ROOT import gROOT, gStyle, gPad, TLegend, TLine, TRandom3
from ROOT import TMVA

batchMode = False

lifetime = float(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(11);

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



