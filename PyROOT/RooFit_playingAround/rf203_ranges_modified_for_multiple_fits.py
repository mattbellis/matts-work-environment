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
#gSystem.Load('libRooFit')

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

#for n in range(0,1000):
for n in range(0,1):
    
    # Generated 10000 events in (x,y) from p.d.f. model
    modelData = model.generate(RooArgSet(x),10000) # RooDataSet
    #################################################################
    # Trying to explicitly give a different name to each dataset!
    #################################################################
    newname = "modelData_%d" % (n)
    modelData.SetName(newname)

    # F i t   f u l l   r a n g e 
    # ---------------------------

    # Fit p.d.f to all data
    r_full = model.fitTo(modelData,RooFit.Save(kTRUE),RooFit.PrintLevel(-1)) # RooFitResult


    # F i t   p a r t i a l   r a n g e 
    # ----------------------------------

    # Define "signal" range in x as [-3,3]
    #####################################################################
    # Trying to explicitly give a different name to each fitting range!
    #####################################################################
    fitrange_name = "signal_%d" % (n)
    x.setRange(fitrange_name,-3,3)   

    # Fit p.d.f only to data in "signal" range
    r_sig = model.fitTo(modelData,RooFit.Save(kTRUE),RooFit.Range(fitrange_name),RooFit.PrintLevel(-1)) # RooFitResult

    # Print fit results 
    print "result of fit on all data "
    r_full.Print()   
    print "result of fit in in signal region (note increased error on signal fraction)" 
    r_sig.Print() 


