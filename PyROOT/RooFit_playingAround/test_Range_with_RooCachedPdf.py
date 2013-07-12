#!/usr/bin/env python

################################################################################
#
# 'ADDITION AND CONVOLUTION' RooFit tutorial macro #203
# 
# Fitting and plotting in sub ranges
#
#
# 07/2008 - Wouter Verkerke 
#
################################################################################

import sys
import ROOT
from ROOT import *
gSystem.Load('libRooFit')

# S e t u p   m o d e l 
# ---------------------

# Construct observables x
x = RooRealVar("x","x",-10,10) 

# Construct gaussx(x,mx,1)
mx = RooRealVar("mx","mx",0,-10,10) 
gx = RooGaussian("gx","gx",x,mx,RooFit.RooConst(1)) 

# Construct px = 1 (flat in x)
px = RooPolynomial("px","px",x) 

# Construct model = f*gx + (1-f)px
f = RooRealVar("f","f",0.,1.) 
model = RooAddPdf("model","model",RooArgList(gx,px),RooArgList(f)) 

# Define "signal" range in x as [-3,3]
x.setRange("signal",-3,3)   

# Generated 10000 events in (x,y) from p.d.f. model
modelData = model.generate(RooArgSet(x),10000) # RooDataSet

cached_model = RooCachedPdf("cmod", "A cached model", model)

cachedhist = cached_model.getCacheHist(RooArgSet(x))

# RooHistPdf
new_model = RooHistPdf("rhp", "RooHistPdf object", RooArgSet(x), cachedhist)

#new_model = RooHistPdf("new_model","Created from cached",cached_model)

#cmodelData = cached_model.generate(RooArgSet(x),10000) # RooDataSet
#cmodelData = cached_model.generate(RooArgSet(x),10000,RooFit.Range("signal")) # RooDataSet
#cmodelData = new_model.generate(RooArgSet(x),10000,RooFit.Range("signal")) # RooDataSet
x.setRange(-3,3)
cmodelData = model.generate(RooArgSet(x),10000) # RooDataSet

frame = x.frame(RooFit.Title("Plotting data a sub range")) # RooPlot
cmodelData.plotOn(frame)
frame.Draw()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

