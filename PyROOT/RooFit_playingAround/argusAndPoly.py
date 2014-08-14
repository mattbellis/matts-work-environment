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
from ROOT import RooAbsPdf, RooProdPdf, RooPolynomial
from ROOT import TCanvas, gStyle, gPad, TH2
from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import gROOT, gStyle

from color_palette import *


#### Command line variables ####
batchMode = False

numevents = int(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)
set_palette("palette",100)


# Build two PDFs
x = RooRealVar("x","x",5.2,5.3) 
y = RooRealVar("y","y",-0.2,0.2) 

# Build polynomial background
p1 = RooRealVar("poly1","1st order coefficient for polynomial",-0.5) 
rarglist = RooArgList(p1)
polyy = RooPolynomial("polyy","Polynomial PDF",y, rarglist);

# Build Argus background PDF
argpar = RooRealVar("argpar","argus shape parameter",-20.0)
cutoff = RooRealVar("cutoff","argus cutoff",5.29)
argus = RooArgusBG("argus","Argus PDF",x,cutoff,argpar)


# Multiply the components
prod = RooProdPdf("bkgd","argus*polyy",RooArgList(argus,polyy)) 

# Generate a toyMC sample
data = prod.generate(RooArgSet(x,y), numevents) # RooDataSet

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 1200, 800)
c.Divide(2,2)

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

# Plot the 2D PDF and the generated data
rllist = RooLinkedList()
rllist.Add(RooFit.Binning(50))
rllist.Add(RooFit.YVar(y, RooFit.Binning(50)))

c.cd(3)
h2_0 = x.createHistogram("x vs y pdf",  rllist)
prod.fillHistogram(h2_0, RooArgList(x,y))
h2_0.Draw("SURF")

c.cd(4)
h2_1 = x.createHistogram("x vs y data",  rllist)
data.fillHistogram(h2_1, RooArgList(x,y))
h2_1.Draw("LEGO")
print h2_1

p1.setConstant(kFALSE)
argpar.setConstant(kFALSE)
cutoff.setConstant(kFALSE)
#prod.fitTo(data, "mh")

gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

