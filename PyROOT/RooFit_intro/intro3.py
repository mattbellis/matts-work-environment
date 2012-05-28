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
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooProdPdf
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

# Build two Gaussian PDFs
x = RooRealVar("x","x",-5,5) 
y = RooRealVar("y","y",-5,5) 
meanx = RooRealVar("mean1","mean of gaussian x",2) 
meany = RooRealVar("mean2","mean of gaussian y",-2) 
sigmax = RooRealVar("sigmax","width of gaussian x",1) 
sigmay = RooRealVar("sigmay","width of gaussian y",5) 
gaussx = RooGaussian("gaussx","gaussian PDF",x,meanx,sigmax)   
gaussy = RooGaussian("gaussy","gaussian PDF",y,meany,sigmay)   

# Multiply the components
prod = RooProdPdf("gaussxy","gaussx*gaussy",RooArgList(gaussx,gaussy)) 

# Generate a toyMC sample
data = prod.generate(RooArgSet(x,y),10000) # RooDataSet

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 900, 500)
c.Divide(2,1)

c.cd(1) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
prod.plotOn(xframe)  # plots f(x) = Int(dy) pdf(x,y)
xframe.Draw() 

c.cd(2) 
yframe = y.frame()  # RooPlot
data.plotOn(yframe, RooLinkedList()) 
prod.plotOn(yframe)  # plots f(y) = Int(dx) pdf(x,y)
yframe.Draw() 

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

