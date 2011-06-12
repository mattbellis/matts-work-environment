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
x = RooRealVar("x","x",-10,10) 
x.setBins(40) 

# Construction a uniform pdf
p0 = RooPolynomial("px","px",x) 

# Sample 1000 events from pdf
data = p0.generate(RooArgSet(x),1000) # RooDataSet



# C a l c u l a t e   w e i g h t   a n d   m a k e   d a t a s e t   w e i g h t e d 
# -----------------------------------------------------------------------------------

# Construct formula to calculate (fake) weight for events
wFunc = RooFormulaVar("w","event weight","(x*x+10)", RooArgList(x)) 

# Add column with variable w to previously generated dataset
w = data.addColumn(wFunc) # RooRealVar

# Dataset d is now a dataset with two observable (x,w) with 1000 entries
data.Print() 

# Instruct dataset wdata in interpret w as event weight rather than as observable
#wdata = RooDataSet(data.GetName(),data.GetTitle(),data,RooArgSet(data.get()),0,w.GetName()) 
wdata = RooDataSet(data.GetName(),data.GetTitle(),data,RooArgSet(data.get()),"",w.GetName())

# Dataset d is now a dataset with one observable (x) with 1000 entries and a sum of weights of ~430K
wdata.Print() 



# U n b i n n e d   M L   f i t   t o   w e i g h t e d   d a t a 
# ---------------------------------------------------------------

# Construction quadratic polynomial pdf for fitting
a0 = RooRealVar("a0","a0",1) 
a1 = RooRealVar("a1","a1",0,-1,1) 
a2 = RooRealVar("a2","a2",1,0,10) 
p2 = RooPolynomial("p2","p2",x,RooArgList(a0,a1,a2),0) 

# Fit quadratic polynomial to weighted data

# NOTE: A plain Maximum likelihood fit to weighted data does in general 
#       NOT result in correct error estimates, unless individual
#       event weights represent Poisson statistics themselves.
#       
# Fit with 'wrong' errors
r_ml_wgt = p2.fitTo(wdata,RooFit.Save()) # RooFitResult

# A first order correction to estimated parameter errors in an 
# (unbinned) ML fit can be obtained by calculating the
# covariance matrix as
#
#    V' = V C-1 V
#
# where V is the covariance matrix calculated from a fit
# to -logL = - sum [ w_i log f(x_i) ] and C is the covariance
# matrix calculated from -logL' = -sum [ w_i^2 log f(x_i) ] 
# (i.e. the weights are applied squared)
#
# A fit in this mode can be performed as follows:

r_ml_wgt_corr = p2.fitTo(wdata,RooFit.Save(),RooFit.SumW2Error(kTRUE)) # RooFitResult



# P l o t   w e i g h e d   d a t a   a n d   f i t   r e s u l t 
# ---------------------------------------------------------------

# Construct plot frame
frame = x.frame(RooFit.Title("Unbinned ML fit, binned chi^2 fit to weighted data")) # RooPlot

# Plot data using sum-of-weights-squared error rather than Poisson errors
wdata.plotOn(frame,RooFit.DataError(RooAbsData.SumW2)) 

# Overlay result of 2nd order polynomial fit to weighted data
p2.plotOn(frame) 



# M L  F i t   o f   p d f   t o   e q u i v a l e n t  u n w e i g h t e d   d a t a s e t
# -----------------------------------------------------------------------------------------

# Construct a pdf with the same shape as p0 after weighting
genPdf = RooGenericPdf("genPdf","x*x+10",RooArgList(x)) 

# Sample a dataset with the same number of events as data
data2 = genPdf.generate(RooArgSet(x),1000) # RooDataSet

# Sample a dataset with the same number of weights as data
data3 = genPdf.generate(RooArgSet(x),43000) # RooDataSet

# Fit the 2nd order polynomial to both unweighted datasets and save the results for comparison
r_ml_unw10 = p2.fitTo(data2,RooFit.Save()) # RooFitResult
r_ml_unw43 = p2.fitTo(data3,RooFit.Save()) # RooFitResult


# C h i 2   f i t   o f   p d f   t o   b i n n e d   w e i g h t e d   d a t a s e t
# ------------------------------------------------------------------------------------

# Construct binned clone of unbinned weighted dataset
binnedData = wdata.binnedClone() # RooDataHist
binnedData.Print("v") 

print " ---------------- "
print " ---------------- "
print " ---------------- "
print " ---------------- "
# Perform chi2 fit to binned weighted dataset using sum-of-weights errors
# 
# NB: Within the usual approximations of a chi2 fit, a chi2 fit to weighted
# data using sum-of-weights-squared errors does give correct error
# estimates
chi2 = RooChi2Var("chi2","chi2",p2,binnedData,RooFit.DataError(RooAbsData.SumW2)) 
m = RooMinuit(chi2) 
#m.migrad() 
m.hesse() 

# Plot chi^2 fit result on frame as well
r_chi2_wgt = m.save() # RooFitResult
p2.plotOn(frame,RooFit.LineStyle(kDashed),RooFit.LineColor(kRed)) 



# C o m p a r e   f i t   r e s u l t s   o f   c h i 2 , M L   f i t s   t o   ( u n ) w e i g h t e d   d a t a 
# ---------------------------------------------------------------------------------------------------------------

# Note that ML fit on 1Kevt of weighted data is closer to result of ML fit on 43Kevt of unweighted data 
# than to 1Kevt of unweighted data, whereas the reference chi^2 fit with SumW2 error gives a result closer to
# that of an unbinned ML fit to 1Kevt of unweighted data. 

print "==> ML Fit results on 1K unweighted events" 
r_ml_unw10.Print() 
print "==> ML Fit results on 43K unweighted events" 
r_ml_unw43.Print() 
print "==> ML Fit results on 1K weighted events with a summed weight of 43K" 
r_ml_wgt.Print() 
print "==> Corrected ML Fit results on 1K weighted events with a summed weight of 43K" 
r_ml_wgt_corr.Print() 
print "==> Chi2 Fit results on 1K weighted events with a summed weight of 43K" 
r_chi2_wgt.Print() 


c = TCanvas("rf403_weightedevts","rf403_weightedevts",600,600) 
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

