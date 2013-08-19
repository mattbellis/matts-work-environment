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
from ROOT import RooAbsPdf, RooProdPdf, RooPolynomial, RooCBShape, RooAbsReal, RooCategory
from ROOT import TCanvas, gStyle, gPad, TH2
from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import gROOT, gStyle

from color_palette import *


#### Command line variables ####
batchMode = False
doFit = False

filename = sys.argv[1]

arglength  = len(sys.argv) - 2
if arglength >= 2:
  if (sys.argv[2] == "doFits"):
    doFits = True

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################

# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)
#set_palette("palette",100)


# Build two PDFs
x = RooRealVar("x","x",5.2,5.3) 
y = RooRealVar("y","y",-0.2,0.2) 
c = RooCategory ("c","c")
c.defineType("Plus",+1)
c.defineType("Minus",-1)

data = RooDataSet("data","data",RooArgSet(x,y,c)) 

infile = open(filename)

i=0
for line in infile.readlines():
  print line
  x.setVal(float(line.split()[0]))
  y.setVal(float(line.split()[1]))
  if i%2==0:
    c.setLabel("Plus")
  else:
    c.setLabel("Minus") 
  data.add(RooArgSet(x,y,c))
  i += 1

#sys.exit(1)

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

sigfrac = RooRealVar("sigfrac","fraction of signal", 0.2)

total = RooAddPdf("total","sig + bkgd",RooArgList(sigProd, bkgdProd), RooArgList(sigfrac))

# Generate a toyMC sample


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
  # Run the fit
  total.fitTo(data, "mh")

rllist = RooLinkedList()
rllist.Add(RooFit.Binning(50))
rllist.Add(RooFit.YVar(y, RooFit.Binning(50)))

c.cd(3)
h2_0 = x.createHistogram("x vs y pdf",  rllist)
total.fillHistogram(h2_0, RooArgList(x,y))
h2_0.Draw("SURF")

c.cd(4)
h2_1 = x.createHistogram("x vs y data",  rllist)
data.fillHistogram(h2_1, RooArgList(x,y))
#h2_1.SetMaximum(5)
h2_1.Draw("BOX")
print h2_1

#c.cd(3) 
#xframe = x.frame() # RooPlot
#data.plotOn(xframe, RooLinkedList()) 
#total.plotOn(xframe)  # plots f(x) = Int(dy) pdf(x,y)
#
#argset = RooArgSet(bkgdProd)
#total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(2))
#argset = RooArgSet(sigProd)
#total.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(3))

#xframe.Draw() 

gPad.Update()


#c.cd(4) 
#yframe = y.frame()  # RooPlot
#data.plotOn(yframe, RooLinkedList()) 
#total.plotOn(yframe)  # plots f(y) = Int(dx) pdf(x,y)

#argset = RooArgSet(bkgdProd)
#total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(2))
#argset = RooArgSet(sigProd)
#total.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(3))

#yframe.Draw() 
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


