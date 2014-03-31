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
from ROOT import RooFormulaVar, RooGenericPdf
from ROOT import TCanvas, gStyle, gPad, TH2
from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import gROOT, gStyle

from color_palette import *


#### Command line variables ####
batchMode = False
doFit = False

filename = sys.argv[1]

arglength  = len(sys.argv) - 1
if arglength >= 2:
  if (sys.argv[2] == "doFit"):
    doFit = True

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

nevents=0
for line in infile.readlines():
  #print line
  x.setVal(float(line.split()[0]))
  y.setVal(float(line.split()[1]))
  if nevents%2==0:
    c.setLabel("Plus")
  else:
    c.setLabel("Minus") 
  data.add(RooArgSet(x,y,c))
  nevents += 1

# Plot data and PDF overlaid
c = TCanvas("c","c",10, 10, 1200, 800)
c.Divide(2,2)

c.cd(1) 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
xframe.Draw() 
gPad.Update()

c.cd(2) 
yframe = y.frame()  # RooPlot
data.plotOn(yframe, RooLinkedList()) 
yframe.Draw() 
gPad.Update()

# Plot the 2D PDF and the generated data
rllist = RooLinkedList()
rllist.Add(RooFit.Binning(50))
rllist.Add(RooFit.YVar(y, RooFit.Binning(50)))

c.cd(3)
h2_0 = x.createHistogram("x vs y pdf",  rllist)
data.fillHistogram(h2_0, RooArgList(x,y))
h2_0.Draw("LEGO")

c.cd(4)
h2_1 = x.createHistogram("x vs y data",  rllist)
data.fillHistogram(h2_1, RooArgList(x,y))
#h2_1.SetMaximum(5)
h2_1.Draw("BOX")

gPad.Update()

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

########################################
# Try running a fit
########################################
if doFit:
  # Make sure they can vary in the fit
  p1.setConstant(kFALSE)
  argpar.setConstant(kFALSE)
  cutoff.setConstant(kFALSE)
  # Set them to be different
  p1.setVal(-1.0)
  argpar.setVal(-20.0)
  # Run the fit

  #########################################################
  # Try setting a range 
  #########################################################
  x.setRange("BKGD1",5.2, 5.30)
  y.setRange("BKGD1",-0.2, -0.1)

  x.setRange("BKGD2",5.2, 5.30)
  y.setRange("BKGD2",0.1, 0.2)

  x.setRange("BKGD3",5.2, 5.26)
  y.setRange("BKGD3",-0.1, 0.1)

  x.setRange("FULL",5.2, 5.30)
  y.setRange("FULL",-0.2, 0.2)

  blind = RooFormulaVar ("blind","blind","(x<5.26)||(y<-0.1||y>0.1)", RooArgList(x,y,y))
  blind = RooFormulaVar ("blind","blind","(x<5.26)", RooArgList(x))
  blindbkgd = RooGenericPdf ("blindbkgd", "bkgdProd*blind", "bkgd*blind", RooArgList(bkgdProd,blind))

  # Run the fit
  #bkgdProd.fitTo(data, "mh")
  rllist = RooLinkedList()
  rllist.Add(RooFit.Range("BKGD1,BKGD2,BKGD3"))
  rllist.Add(RooFit.NormRange("BKGD1,BKGD2,BKGD3"))
  rllist.Add(RooFit.FitOptions("mh"))
  bkgdProd.fitTo(data,rllist)
  #blindbkgd.fitTo(data,rllist)

  print "Finished the fit!!!!!!!!!!!\n\n"

  #########################################################
  # Generate MC data over the bkgdProd range
  #########################################################
  #rllist_0 = RooLinkedList()
  #rllist_0.Add(RooFit.Range("BKGD1,BKGD2,BKGD3"))
  mcdata_all = bkgdProd.generate(RooArgSet(x,y), nevents) # RooDataSet
  print "A\n"
  mcdata1 = mcdata_all.reduce(RooFit.CutRange("BKGD1"))
  mcdata1.append(mcdata_all.reduce(RooFit.CutRange("BKGD3")))
  mcdata1.append(mcdata_all.reduce(RooFit.CutRange("BKGD2")))
  print "B\n"

  c.cd(1) 
  rllist_mc = RooLinkedList()
  rllist_mc.Add(RooFit.MarkerColor(2))
  #rllist_mc.Add(RooFit.Range("BKGD1"))
  #rllist_mc.Add(RooFit.NormRange("BKGD1"))
  mcdata1.plotOn(xframe, rllist_mc)
  xframe.Draw("same") 
  #mcdata2.plotOn(xframe, rllist_mc)
  #xframe.Draw("same") 
  #mcdata3.plotOn(xframe, rllist_mc)
  #xframe.Draw("same") 
  gPad.Update()

  c.cd(2) 
  mcdata1.plotOn(yframe, rllist_mc)
  yframe.Draw("same") 
  #mcdata2.plotOn(yframe, rllist_mc)
  #yframe.Draw("same") 
  #mcdata3.plotOn(yframe, rllist_mc)
  #yframe.Draw("same") 
  gPad.Update()


  print "C\n"

  ############################
  c.cd(1)
  argset = RooArgSet(bkgdProd)
  bkgdProd.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(2), RooFit.Range("FULL"), RooFit.NormRange("BKGD1,BKGD2,BKGD3"))
  #bkgdProd.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(2))
  xframe.Draw()
  gPad.Update()

  argset = RooArgSet(blindbkgd)
  blindbkgd.plotOn(xframe, RooFit.Components(argset), RooFit.LineColor(4), RooFit.LineStyle(2))
  xframe.Draw()
  gPad.Update()

  ############################
  c.cd(2)
  argset = RooArgSet(bkgdProd)
  bkgdProd.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(2), RooFit.Range("FULL"), RooFit.NormRange("BKGD1,BKGD2,BKGD3"))
  #bkgdProd.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(2))
  yframe.Draw()
  gPad.Update()

  argset = RooArgSet(blindbkgd)
  blindbkgd.plotOn(yframe, RooFit.Components(argset), RooFit.LineColor(4), RooFit.LineStyle(2))
  yframe.Draw()
  gPad.Update()






## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


