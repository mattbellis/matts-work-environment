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
#from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
#from ROOT import RooAbsPdf, RooProdPdf, RooPolynomial
#from ROOT import TCanvas, gStyle, gPad, TH2
#from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
#from ROOT import gROOT, gStyle

from color_palette import *


#### Command line variables ####
batchMode = False

#numevents = int(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################
# C r e a t e   3 D   p d f   a n d   d a t a 
# -------------------------------------------

# Create observables
#x = RooRealVar ("x","x",-5,5) 
x = RooRealVar ("x","x",-0.2,0.2) 

meanCB = RooRealVar("mCB","m of gaussian of CB", 0.00)
sigmaCB = RooRealVar("sigmaCB","width of gaussian in CB", 0.010)
alphaCB = RooRealVar("alphaCB", "alpha of CB", 0.5)
nCB = RooRealVar("nCB","n of CB", 1.0)
cb = RooCBShape("gauss2", "Crystal Barrel Shape PDF", x, meanCB, sigmaCB, alphaCB, nCB)

# P r o j e c t   p d f   a n d   d a t a   o n   x
# -------------------------------------------------

# Make plain projection of data and pdf on x observable
frame = x.frame(RooFit.Title("Projection of 3D data and pdf on X"), RooFit.Bins(40)) 
cb.plotOn(frame) 



c = TCanvas("can","can",10,10, 800,400) 
c.Divide(1,1) 

c.cd(1)  
frame.Draw() 


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

