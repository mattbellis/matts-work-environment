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
doFit = False

numevents = int(sys.argv[1])
fractionamount = float(sys.argv[2])

arglength  = len(sys.argv) - 1
if arglength >= 3:
  if (sys.argv[3] == "doFit"):
    doFit = True

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

#####
# Signal PDF
#####
mean1 = RooRealVar("mean1","mean of gaussian 1", 5.279)
mean2 = RooRealVar("mean2","mean of gaussian 2", 0.000)
sigma1 = RooRealVar("sigma1","width of gaussian 1", 0.005)
sigma2 = RooRealVar("sigma2","width of gaussian 2", 0.030)
gauss1 = RooGaussian("gauss1", "gaussian 1 PDF", x, mean1, sigma1)
gauss2 = RooGaussian("gauss2", "gaussian 2 PDF", y, mean2, sigma2)

sigProd = RooProdPdf("sig","gauss1*gauss2",RooArgList(gauss1, gauss2)) 

# Build polynomial background
p1 = RooRealVar("poly1","1st order coefficient for polynomial",-0.5) 
rarglist = RooArgList(p1)
polyy = RooPolynomial("polyy","Polynomial PDF",y, rarglist);

# Build Argus background PDF
argpar = RooRealVar("argpar","argus shape parameter",-20.0)
cutoff = RooRealVar("cutoff","argus cutoff",5.29)
argus = RooArgusBG("argus","Argus PDF",x,cutoff,argpar)

# Multiply the components
bkgdProd = RooProdPdf("bkgd","argus*polyy",RooArgList(argus,polyy)) 

sigfrac = RooRealVar("sigfrac","fraction of signal", fractionamount)

total = RooAddPdf("total","sig + bkgd",RooArgList(sigProd, bkgdProd), RooArgList(sigfrac))

# Generate a toyMC sample
data = total.generate(RooArgSet(x,y), numevents) # RooDataSet

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 1200, 800)
c.Divide(2,2)

c.cd(1) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
total.plotOn(xframe)  # plots f(x) = Int(dy) pdf(x,y)

argset = RooArgSet(bkgdProd)
total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(2))
argset = RooArgSet(sigProd)
total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(3))

xframe.Draw() 
gPad.Update()


c.cd(2) 
yframe = y.frame()  # RooPlot
data.plotOn(yframe, RooLinkedList()) 
total.plotOn(yframe)  # plots f(y) = Int(dx) pdf(x,y)

argset = RooArgSet(bkgdProd)
total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(2))
argset = RooArgSet(sigProd)
total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(3))

yframe.Draw() 
gPad.Update()

# Plot the 2D PDF and the generated data

########################################
# Try running a fit
########################################
if doFit:
  # Make sure they can vary in the fit
  p1.setConstant(kFALSE)
  argpar.setConstant(kFALSE)
  cutoff.setConstant(kFALSE)
  mean1.setConstant(kFALSE)
  mean2.setConstant(kFALSE)
  sigma1.setConstant(kFALSE)
  sigma2.setConstant(kFALSE)
  sigfrac.setConstant(kFALSE)
  # Set them to be different
  p1.setVal(-1.0)
  argpar.setVal(-20.0)
  cutoff.setVal(5.280)
  mean1.setVal(5.279)
  mean2.setVal(0.0)
  sigma1.setVal(0.010)
  sigma2.setVal(0.010)
  sigfrac.setVal(0.20)
  # Run the fit
  total.fitTo(data, "mh")

c.cd(3) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
total.plotOn(xframe)  # plots f(x) = Int(dy) pdf(x,y)

argset = RooArgSet(bkgdProd)
total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(2))
argset = RooArgSet(sigProd)
total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(3))

xframe.Draw() 
gPad.Update()


c.cd(4) 
yframe = y.frame()  # RooPlot
data.plotOn(yframe, RooLinkedList()) 
total.plotOn(yframe)  # plots f(y) = Int(dx) pdf(x,y)

argset = RooArgSet(bkgdProd)
total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(2))
argset = RooArgSet(sigProd)
total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(3))

yframe.Draw() 
gPad.Update()

# Plot the 2D PDF and the generated data
#numsig = sigProd.getAnalyticalIntegralWN()
#print numsig


gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

