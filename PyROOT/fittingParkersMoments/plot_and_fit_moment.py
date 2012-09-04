#!/usr/bin/env python

################################################################################
#
# 'DATA AND CATEGORIES' RooFit tutorial macro #403
# 
# Using weights in unbinned datasets
#
#
#
# 07/2008 - Wouter Verkerke 
# 
################################################################################

import sys
import ROOT
from ROOT import *

# C r e a t e   o b s e r v a b l e   a n d   u n w e i g h t e d   d a t a s e t 
# -------------------------------------------------------------------------------

# Declare observable
#x = RooRealVar("x","x",0.64,1.31) 
x = RooRealVar("x","x",1.05,1.715) 
x.setBins(140) 

w = RooRealVar("w","w",-2,2) 

# Create the dataset
data = RooDataSet("data","data",RooArgSet(x,w),RooFit.WeightVar(w))

# Read in the data
inputfilename = sys.argv[1]
infile = open(inputfilename)

for line in infile:
    print line
    x.setVal(float(line.split()[0]))
    w.setVal(float(line.split()[1]))
    data.add(RooArgSet(x,w))

# Dataset d is now a dataset with one observable (x) and the weight (w)
data.Print() 

# U n b i n n e d   M L   f i t   t o   w e i g h t e d   d a t a 
# ---------------------------------------------------------------

# Construction quadratic polynomial pdf for fitting
mean = RooRealVar("mean","mean",0.896,0.6,1.4) 
sigma = RooRealVar("sigma","sigma",0.07,0.01,1.0)

a0 = RooRealVar("a0","a1",0.9)
a1 = RooRealVar("a1","a1",0)
a2 = RooRealVar("a2","a2",-0.4)
a3 = RooRealVar("a3","a3",0.1)

a0.setConstant(kFALSE)
a1.setConstant(kFALSE)
a2.setConstant(kFALSE)
a3.setConstant(kFALSE)

p2 = RooPolynomial("p2","p2",x,RooArgList(a0,a1,a2,a3),0) 
gaus = RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

fraction = RooRealVar("fraction","fraction of component 1 in signal",0.8,0.,1.) 
total = RooAddPdf("total","total",RooArgList(p2,gaus),RooArgList(fraction)) 


# Fit quadratic polynomial to weighted data
binned_data = data.binnedClone() # RooDataHist

rllist = RooLinkedList()
rllist.Add(RooFit.Save())
rllist.Add(RooFit.Strategy(2))
r_chi2_wgt = total.chi2FitTo(binned_data,rllist) # RooFitResult


# P l o t   w e i g h e d   d a t a   a n d   f i t   r e s u l t 
# ---------------------------------------------------------------

# Construct plot frame
frame = x.frame(RooFit.Title("Unbinned ML fit, binned chi^2 fit to weighted data")) # RooPlot

# Plot data using sum-of-weights-squared error rather than Poisson errors
#data.plotOn(frame,RooFit.DataError(RooAbsData.SumW2)) 
data.plotOn(frame)

# Overlay result of 2nd order polynomial fit to weighted data
total.plotOn(frame) 

r_chi2_wgt.Print() 

c = TCanvas("can","can",600,600) 
gPad.SetLeftMargin(0.15)  
frame.GetYaxis().SetTitleOffset(1.8) 
frame.Draw() 
gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

