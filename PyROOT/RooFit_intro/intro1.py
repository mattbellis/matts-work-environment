#!/usr/bin/env python

###############################################################
# intro1.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro1.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList
from ROOT import gStyle

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

# Elementary operations on a gaussian PDF
# Build Gaussian PDF
x = RooRealVar("x","x",-10,10) 
mean = RooRealVar("mean","mean of gaussian",-1) 
sigma = RooRealVar("sigma","width of gaussian",3) 
gauss = RooGaussian("gauss","gaussian PDF",x,mean,sigma)   

# Generate a toy MC set
data = gauss.generate(RooArgSet(x), 10000) # RooDataSet

# Fit pdf to toy
mean.setConstant(kFALSE) 
sigma.setConstant(kFALSE) 

# Start the fit parameters far from the known value,
# to test robustness of minimizer.
mean.setVal(-20.0)
sigma.setVal(10.0)
gauss.fitTo(data, 'mh') 
# Available fit options:
#  "m" = MIGRAD only, i.e. no MINOS
#  "s" = estimate step size with HESSE before starting MIGRAD
#  "h" = run HESSE after MIGRAD
#  "e" = Perform extended MLL fit
#  "0" = Run MIGRAD with strategy MINUIT 0 (no correlation matrix calculation at end)
#        Does not apply to HESSE or MINOS, if run afterwards.
#  "q" = Switch off verbose mode
#  "l" = Save log file with parameter values at each MINUIT step
#  "v" = Show changed parameters at each MINUIT step
#  "t" = Time fit
#  "r" = Save fit output in RooFitResult object (return value is object RFR pointer)
# Available optimizer options
#  "c" = Cache and precalculate components of PDF that exclusively depend on constant parameters
#  "2" = Do NLL calculation in multi-processor mode on 2 processors
#  "3" = Do NLL calculation in multi-processor mode on 3 processors
#  "4" = Do NLL calculation in multi-processor mode on 4 processors

# Plot PDF and toy data overlaid
xframe2 = x.frame() # RooPlot
data.plotOn(xframe2, RooLinkedList() ) 
gauss.plotOn(xframe2) 
xframe2.Draw() 

# Print final value of parameters
mean.Print() 
sigma.Print() 

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

