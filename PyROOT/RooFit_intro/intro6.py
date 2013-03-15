#!/usr/bin/env python

###############################################################
# intro6.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro6.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar, RooCategory, RooRealConstant, RooSuperCategory
from ROOT import RooMappedCategory, RooThresholdCategory, RooTruthModel, RooDecay, RooGaussModel
from ROOT import RooAddModel
from ROOT import gStyle
from ROOT import TCanvas

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

# Convoluted PDFs
# Build a simple decay PDF
dt = RooRealVar("dt","dt",-20,20) 
tau = RooRealVar("tau","tau",1.548) 

# Build a truth resolution model (delta function)
tm = RooTruthModel("tm","truth model",dt) 

# Construct a simple unsmeared decay PDF
decay_tm = RooDecay("decay_tm","decay",dt,tau,tm,RooDecay.DoubleSided) 

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 1200, 400)
c.Divide(3,1)
c.cd(1) 
decay_tm.plotOn(dt.frame()).Draw() 

# Build a gaussian resolution model
bias1 = RooRealVar("bias1","bias1",0) 
sigma1 = RooRealVar("sigma1","sigma1",1)   
gm1 = RooGaussModel("gm1","gauss model 1",dt,bias1,sigma1) 

# Construct a decay PDF, smeared with single gaussian resolution model
decay_gm1 = RooDecay("decay_gm1","decay",dt,tau,gm1,RooDecay.DoubleSided) 

c.cd(2) 
decay_gm1.plotOn(dt.frame()).Draw() 

# Build another gaussian resolution model
bias2 = RooRealVar("bias2","bias2",0) 
sigma2 = RooRealVar("sigma2","sigma2",5)   
gm2 = RooGaussModel("gm2","gauss model 2",dt,bias2,sigma2) 

# Build a composite resolution model
gm1frac = RooRealVar("gm1frac","fraction of gm1",0.5) 
gmsum = RooAddModel("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2), RooArgList(gm1frac)) 

# Construct a decay PDF, smeared with double gaussian resolution model
decay_gmsum = RooDecay("decay_gmsum","decay",dt,tau,gmsum,RooDecay.DoubleSided) 

c.cd(3) 
decay_gmsum.plotOn(dt.frame()).Draw() 

##########################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

