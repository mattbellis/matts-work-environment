#!/usr/bin/env python

###############################################################
# intro8.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro8.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from math import *
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, kTRUE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar, RooCategory, RooRealConstant, RooSuperCategory
from ROOT import RooMappedCategory, RooThresholdCategory, RooTruthModel, RooDecay, RooGaussModel
from ROOT import RooAddModel, RooDataHist
from ROOT import TCanvas, TFile, gPad, gStyle

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
##########################################################

# Some global style settings
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

# Dataset operations
# Binned (RooDataHist) and unbinned datasets (RooDataSet) share
# many properties and inherit from a common abstract base class
# (RooAbsData), that provides an interface for all operations
# that can be performed regardless of the data format
x = RooRealVar("x","x",-10,10) 
y = RooRealVar("y","y", 0, 40) 
c = RooCategory("c","c") 
c.defineType("Plus",+1) 
c.defineType("Minus",-1) 

# *** Unbinned datasets ***

# RooDataSet is an unbinned dataset (a collection of points in N-dimensional space)
d = RooDataSet("d","d", RooArgSet(x,y,c)) 

# Unlike RooAbsArgs (RooAbsPdf,RooFormulaVar,....) datasets are not attached to 
# the variables they are constructed from. Instead they are attached to an internal 
# clone of the supplied set of arguments

# Fill d with dummy values
for i in range(0,1000):
  x.setVal(i/50 - 10 )
  y.setVal(sqrt(1.0*i)) 
  if i%2:
    c.setLabel("Plus")
  else:
    c.setLabel("Minus")

  # We must explicitly refer to x,y,c here to pass the values because
  # d is not linked to them (as explained above)
  d.add(RooArgSet(x,y,c)) 
# End loop

d.Print("v") 
print("\n")

# The get() function returns a pointer to the internal copy of the RooArgSet(x,y,c)
# supplied in the constructor
row = d.get() # RooArgSet 
row.Print("v") 
print("\n")

# Get with an argument loads a specific data point in row and returns
# a pointer to row argset. get() always returns the same pointer, unless
# an invalid row number is specified. In that case a null ptr is returned
d.get(900).Print("v") 
print("\n")

# *** Reducing / Appending / Merging ***

# The reduce() function returns a new dataset which is a subset of the original
print("\n>> d1 has only columns x,c") 
d1 = d.reduce(RooArgSet(x,c)) # RooDataSet 
d1.Print("v") 

print("\n>> d2 has only column y") 
d2 = d.reduce(RooArgSet(y)) # RooDataSet 
d2.Print("v") 

print("\n>> d3 has only the points with y>5.17") 
d3 = d.reduce("y>5.17") # RooDataSet 
d3.Print("v") 

print("\n>> d4 has only columns x,c for data points with y>5.17") 
d4 = d.reduce(RooArgSet(x,c),"y>5.17")  # RooDataSet 
d4.Print("v") 

# The merge() function adds two data set column-wise
print("\n>> merge d2(y) with d1(x,c) to form d1(x,c,y)") 
d1.merge(d2)  
d1.Print("v") 

# The append() function addes two datasets row-wise
print("\n>> append data points of d3 to d1") 
d1.append(d3) 
d1.Print("v")   

# *** Binned datasets ***

# A binned dataset can be constructed empty, from an unbinned dataset, or
# from a ROOT native histogram (TH1,2,3)

print(">> construct dh (binned) from d(unbinned) but only take the x and y dimensions,") 
print(">> the category 'c' will be projected in the filling process") 

# The binning of real variables (like x,y) is done using their fit range
#get/setFitRange()' and number of specified fit bins 'get/setFitBins()'.
# Category dimensions of binned datasets get one bin per defined category state
x.setBins(10) 
y.setBins(10) 
dh = RooDataHist("dh","binned version of d",RooArgSet(x,y),d) 
dh.Print("v") 

yframe = y.frame(10) # RooPlot
dh.plotOn(yframe, RooLinkedList() )  # plot projection of 2D binned data on y
#RooDataHist.plotOn(dh,yframe) 
#super(RooTreeData,dh).plotOn(yframe)
yframe.Draw() 

# Examine the statistics of a binned dataset
print(">> number of bins in dh   : " + str(dh.numEntries())) 
print(">> sum of weights in dh   : " + str(dh.sum(kFALSE))) 
print(">> integral over histogram: " + str(dh.sum(kTRUE))) # accounts for bin volume

# Locate a bin from a set of coordinates and retrieve its properties
x.setVal(0.3)
y.setVal(20.5)
print(">> retrieving the properties of the bin enclosing coordinate (x,y) = (0.3,20.5) ") 
print(" bin center:") 
dh.get(RooArgSet(x,y)).Print("v")  # load bin center coordinates in internal buffer
print(" weight = " + str(dh.weight())) # return weight of last loaded coordinates

# Reduce the 2-dimensional binned dataset to a 1-dimensional binned dataset
#
# All reduce() methods are interfaced in RooAbsData. All reduction techniques
# demonstrated on unbinned datasets can be applied to binned datasets as well.
print(">> Creating 1-dimensional projection on y of dh for bins with x>0") 
dh2 = dh.reduce( RooArgSet(y), "x>0") # RooDataHist 
dh2.Print("v") 

# Add dh2 to yframe and redraw
dh2.plotOn(yframe, RooFit.LineColor(2), RooFit.MarkerColor(2)) 
#super(RooTreeData,dh2).plotOn(yframe, RooFit.LineColor(2), RooFit.MarkerColor(2))
yframe.Draw() 
gPad.Update()

# *** Saving and loading from file ***

# Datasets can be persisted with ROOT I/O
print("\n>> Persisting d via ROOT I/O") 
f = TFile ("intro8.root","RECREATE") 
d.Write() 
f.ls() 


# To read back in future session:
# > TFile f("intro8.root") 
# > RooDataSet* d = (RooDataSet*) f.FindObject("d") 

##########################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

