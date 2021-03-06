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
from ROOT import RooAbsPdf, RooProdPdf, RooPolynomial, RooCBShape, RooAbsReal
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
meandE = RooRealVar("meandE","mean of gaussian dE", 0.000)
sigmadE = RooRealVar("sigmadE","width of gaussian dE", 0.020)
gaussdE = RooGaussian("gaussdE", "gaussian dE PDF", y, meandE, sigmadE)

meanCB = RooRealVar("mCB","m of gaussian of CB", 5.279)
sigmaCB = RooRealVar("sigmaCB","width of gaussian in CB", 0.0028)
alphaCB = RooRealVar("alphaCB", "alpha of CB", 2.0)
nCB = RooRealVar("nCB","n of CB", 1.0)
cb = RooCBShape("gauss2", "Crystal Barrel Shape PDF", x, meanCB, sigmaCB, alphaCB, nCB)

sigProd = RooProdPdf("sig","gaussdE*cb",RooArgList(gaussdE, cb)) 

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
#########################################################
# Try setting a range 
#########################################################
x.setRange("SB1",5.2, 5.26)
y.setRange("SB1",-0.2, -0.1)

data = total.generate(RooArgSet(x,y), numevents, RooFit.Range("SB1")) # RooDataSet

for i in range(0,data.numEntries()):
  data.get(i).Print("v")
  print str(data.get(i).getRealValue("x")) + " " + str(data.get(i).getRealValue("y"))

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 1200, 800)
c.Divide(2,2)

c.cd(1) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
#total.plotOn(xframe,  RooFit.Normalization(1.0,RooAbsReal.RelativeExpected), RooFit.LineColor(2) )
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
  meandE.setConstant(kFALSE)
  sigmadE.setConstant(kFALSE)
  meanCB.setConstant(kFALSE)
  sigmaCB.setConstant(kFALSE)
  alphaCB.setConstant(kFALSE)
  nCB.setConstant(kFALSE)
  p1.setConstant(kFALSE)
  argpar.setConstant(kFALSE)
  cutoff.setConstant(kFALSE)
  sigfrac.setConstant(kFALSE)
  # Set them to be different
  meandE.setVal(0.0)
  sigmadE.setVal(0.050)
  meanCB.setVal(5.279)
  sigmaCB.setVal(0.005)
  alphaCB.setVal(1.0)
  nCB.setVal(1.0)
  p1.setVal(-1.0)
  argpar.setVal(-20.0)
  cutoff.setVal(5.280)
  #########################################################
  # Try setting a range 
  #########################################################
  x.setRange("SB1",5.2, 5.26)
  y.setRange("SB1",-0.2, -0.1)


  # Run the fit
  #total.fitTo(data, "mh")
  total.fitTo(data,RooFit.Range("SB1"))

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

