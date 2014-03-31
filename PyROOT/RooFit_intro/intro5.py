#!/usr/bin/env python

###############################################################
# intro5.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro5.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar, RooCategory, RooRealConstant, RooSuperCategory
from ROOT import RooMappedCategory, RooThresholdCategory
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

# Define a category with explicitly numbered states
b0flav = RooCategory("b0flav","B0 flavour eigenstate") 
b0flav.defineType("B0",-1) 
b0flav.defineType("B0bar",1) 
b0flav.Print("s") 

# Define a category with labels only
tagCat = RooCategory("tagCat","Tagging category") 
tagCat.defineType("Lepton") 
tagCat.defineType("Kaon") 
tagCat.defineType("NetTagger-1") 
tagCat.defineType("NetTagger-2") 
tagCat.Print("s") 

# Define a dummy PDF in x 
x = RooRealVar("x","x",0,10) 
a = RooArgusBG("a", "argus(x)", x, RooRealConstant.value(10), RooRealConstant.value(-1)) 

# Generate a dummy dataset 
data = a.generate(RooArgSet(x, b0flav, tagCat), 10000) # RooDataSet

# Tables are equivalent of plots for categories
btable = data.table(b0flav) # RooTable
btable.Print() 
ttable = data.table(tagCat, "x>8.23")  # RooTable
ttable.Print() 

# Super-category is 'product' of categories
b0Xtcat = RooSuperCategory("b0Xtcat", "b0flav X tagCat", RooArgSet(b0flav, tagCat)) 
bttable = data.table(b0Xtcat) # RooTable 
bttable.Print() 

# Mapped category is category.category function
tcatType = RooMappedCategory("tcatType", "tagCat type", tagCat, "Unknown") 
tcatType.map("Lepton", "Cut based") 
tcatType.map("Kaon", "Cut based") 
tcatType.map("NetTagger*", "Neural Network") 
(data.table(tcatType)).Print() 

# Threshold category is real.category function
xRegion = RooThresholdCategory("xRegion", "region of x", x, "Background") 
xRegion.addThreshold(4.23, "Background") 
xRegion.addThreshold(5.23, "SideBand") 
xRegion.addThreshold(8.23, "Signal") 
xRegion.addThreshold(9.23, "SideBand")  
#
# Background | SideBand | Signal | SideBand | Background
#           4.23       5.23     8.23       9.23 
data.addColumn(xRegion) 
xframe = x.frame() # RooPlot 
data.plotOn(xframe, RooLinkedList()) 
rlist = RooLinkedList()
rlist.Add(RooFit.Cut("xRegion==xRegion::SideBand"))
rlist.Add(RooFit.MarkerColor(2))
rlist.Add(RooFit.MarkerSize(2))
data.plotOn(xframe, rlist)
xframe.Draw() 


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

