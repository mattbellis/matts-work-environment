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
from ROOT import *
#from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
#from ROOT import RooAbsPdf, RooProdPdf, RooPolynomial
#from ROOT import TCanvas, gStyle, gPad, TH2
#from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
#from ROOT import gROOT, gStyle

from color_palette import *


#### Command line variables ####
batchMode = False

numevents = int(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
################################################
# C r e a t e   3 D   p d f   a n d   d a t a 
# -------------------------------------------

# Create observables
x = RooRealVar ("x","x",-5,5) 
y = RooRealVar ("y","y",-5,5) 
z = RooRealVar ("z","z",-5,5) 

# Create signal pdf gauss(x)*gauss(y)*gauss(z) 
gx = RooGaussian ("gx","gx",x,RooFit.RooConst(0),RooFit.RooConst(1)) 
gy = RooGaussian ("gy","gy",y,RooFit.RooConst(0),RooFit.RooConst(1)) 
gz = RooGaussian ("gz","gz",z,RooFit.RooConst(0),RooFit.RooConst(1)) 
sig = RooProdPdf ("sig","sig", RooArgList(gx,gy,gz)) 

# Create background pdf poly(x)*poly(y)*poly(z) 
px = RooPolynomial ("px","px",x,RooArgList(RooFit.RooConst(-0.1),RooFit.RooConst(0.004))) 
py = RooPolynomial ("py","py",y,RooArgList(RooFit.RooConst(0.1),RooFit.RooConst(-0.004))) 
pz = RooPolynomial ("pz","pz",z) 
bkg = RooProdPdf ("bkg","bkg",RooArgList(px,py,pz)) 

# Create composite pdf sig+bkg
fsig = RooRealVar ("fsig","signal fraction",0.1,0.,1.) 
model = RooAddPdf ("model","model",RooArgList(sig,bkg), RooArgList(fsig)) 

data = model.generate(RooArgSet(x,y,z),20000) 



# P r o j e c t   p d f   a n d   d a t a   o n   x
# -------------------------------------------------

# Make plain projection of data and pdf on x observable
frame = x.frame(RooFit.Title("Projection of 3D data and pdf on X"), RooFit.Bins(40)) 
rllist = RooLinkedList()
rllist.Add(RooFit.MarkerColor(2))
#data.plotOn(frame, RooLinkedList()) 
data.plotOn(frame, rllist)
model.plotOn(frame) 



# P r o j e c t   p d f   a n d   d a t a   o n   x   i n   s i g n a l   r a n g e
# ----------------------------------------------------------------------------------

# Define signal region in y and z observables
x.setRange("sigRegion",-1,1) 
y.setRange("sigRegion",-1,1) 
z.setRange("sigRegion",-1,1) 


# Make plot frame
frame2 = x.frame(RooFit.Title("Same projection on X in signal range of (Y,Z)"), RooFit.Bins(40)) 

# Plot subset of data in which all observables are inside "sigRegion"
# For observables that do not have an explicit "sigRegion" range defined (e.g. observable)
# an implicit definition is used that is identical to the full range (i.e. [-5,5] for x)
print "A"
#data.plotOn(frame2, RooFit.CutRange("sigRegion")) 
rllist = RooLinkedList()
rllist.Add(RooFit.CutRange("sigRegion"))
data.plotOn(frame2, rllist)
print "BA"

# Project model on x, integrating projected observables (y,z) only in "sigRegion"
model.plotOn(frame2, RooFit.ProjectionRange("sigRegion")) 


c = TCanvas("rf311_rangeplot","rf310_rangeplot",800,400) 
c.Divide(2) 

c.cd(1)  
frame.Draw() 

c.cd(2)  
frame2.Draw() 


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

