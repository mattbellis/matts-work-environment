#!/usr/bin/env python

###############################################################
# intro4.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro4.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar
from ROOT import TCanvas, gStyle

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

# Build Gaussian PDF
x = RooRealVar("x","x",-10,10) 
y = RooRealVar("y","y",0,3) 

#  g(x,m,s)  
#  m -> m(y) = m0 + m1*y 
#  g(x,m(y),s)

# Build a parameterized mean variable for gauss
mean0 = RooRealVar("mean0","offset of mean function",0.5) 
mean1 = RooRealVar("mean1","slope of mean function",3.0) 
mean =  RooFormulaVar("mean","parameterized mean","mean0+mean1*y",RooArgList(mean0,mean1,y)) 

sigma = RooRealVar("sigma","width of gaussian",3) 
gauss = RooGaussian("gauss","gaussian PDF",x,mean,sigma)   

# Generate a toy MC set
data = gauss.generate(RooArgSet(x,y),10000)  # RooDataSet 

# Plot data and PDF overlaid
c = TCanvas("c","c", 10, 10, 900, 500)
c.Divide(2,1)

# Plot x projection
c.cd(1) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
gauss.plotOn(xframe)  # plots f(x) = Int(dy) pdf(x,y)
xframe.Draw() 

# Plot y projection
c.cd(2) 
yframe = y.frame() # RooPlot
data.plotOn(yframe, RooLinkedList()) 
gauss.plotOn(yframe)  # plots f(y) = Int(dx) pdf(x,y)
yframe.Draw() 

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

