#!/usr/bin/env python

###############################################################
# intro2.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro2.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf
from ROOT import gStyle, gPad

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
x = RooRealVar("x","x",0,10) 
mean1 = RooRealVar("mean1","mean of gaussian 1",2) 
mean2 = RooRealVar("mean2","mean of gaussian 2",3) 
sigma = RooRealVar("sigma","width of gaussians",1) 
gauss1 = RooGaussian("gauss1","gaussian PDF",x,mean1,sigma)   
gauss2 = RooGaussian("gauss2","gaussian PDF",x,mean2,sigma)   

# Build Argus background PDF
argpar = RooRealVar("argpar","argus shape parameter",-1.0) 
cutoff = RooRealVar("cutoff","argus cutoff",9.0) 
argus = RooArgusBG("argus","Argus PDF",x,cutoff,argpar) 

# Add the components
g1frac = RooRealVar("g1frac","fraction of gauss1",0.5) 
g2frac = RooRealVar("g2frac","fraction of gauss2",0.1) 
sum = RooAddPdf("sum","g1+g2+a",RooArgList(gauss1,gauss2,argus), RooArgList(g1frac,g2frac)) 

# Generate a toyMC sample
data = sum.generate(RooArgSet(x), 10000) # RooDataSet
                                      
# Plot data and PDF overlaid
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList() ) 
sum.plotOn(xframe) 


#sum.plotOn(xframe, RooFit.Components(RooArgSet(argus, gauss2)), RooFit.LineColor(2)) 
argset = RooArgSet(argus, gauss2)
sum.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(ROOT.kRed)) 
#super(RooAddPdf,sum).plotOn(xframe, RooFit.Components(RooArgSet(argus, gauss2)), RooFit.LineColor(2)) 

# Looking at the parameters of a PDF
paramList = sum.getParameters(data) # RooArgSet
paramList.Print("v") 

xframe.Draw() 

gPad.Update()

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

