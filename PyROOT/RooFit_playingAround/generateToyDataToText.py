#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

import backgroundAndSignal_def
from backgroundAndSignal_def import *

#### Command line variables ####
batchMode = False
makeCuts = False

numevents = int(sys.argv[1])
fractionamount = float(sys.argv[2])

arglength  = len(sys.argv) - 1
if arglength >= 3:
  if (sys.argv[3] == "makeCuts"):
    makeCuts = True

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)
set_palette("palette",100)

# Generate a toyMC sample

data = total.generate(RooArgSet(x,y), numevents) # RooDataSet

for i in range(0,data.numEntries()):
  #data.get(i).Print("v")
  cut = True
  x = data.get(i).getRealValue("x")
  y = data.get(i).getRealValue("y")
  if makeCuts:
    cut = x>5.27 and x<5.30
    cut = not(cut and y>-0.075 and y<0.075)

  if cut:
    print str(x) + " " + str(y)
  #print str(data.get(i).getRealValue("x")) + " " + str(data.get(i).getRealValue("y"))


