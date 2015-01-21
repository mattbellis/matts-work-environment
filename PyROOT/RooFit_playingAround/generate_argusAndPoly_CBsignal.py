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

data = total.generate(RooArgSet(x,y), numevents) # RooDataSet

for i in range(0,data.numEntries()):
  #data.get(i).Print("v")
  cut = True
  x = data.get(i).getRealValue("x")
  y = data.get(i).getRealValue("y")
  if makeCuts:
    cut = x>5.26 and x<5.30
    cut = not(cut and y>-0.1 and y<0.1)

  if cut:
    print str(x) + " " + str(y)
  #print str(data.get(i).getRealValue("x")) + " " + str(data.get(i).getRealValue("y"))


