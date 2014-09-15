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

# Generated 10000 events in (x,y) from p.d.f. model
modelData = model.generate(RooArgSet(x),10000) # RooDataSet

# F i t   f u l l   r a n g e 
# ---------------------------

# Fit p.d.f to all data
r_full = model.fitTo(modelData,RooFit.Save(kTRUE)) # RooFitResult


# F i t   p a r t i a l   r a n g e 
# ----------------------------------

# Define "signal" range in x as [-3,3]
x.setRange("signal",-3,3)   

# Fit p.d.f only to data in "signal" range
r_sig = model.fitTo(modelData,RooFit.Save(kTRUE),RooFit.Range("signal")) # RooFitResult


# P l o t   /   p r i n t   r e s u l t s 
# ---------------------------------------

# Make plot frame in x and add data and fitted model
frame = x.frame(RooFit.Title("Fitting a sub range")) # RooPlot
modelData.plotOn(frame) 
model.plotOn(frame,RooFit.Range("Full"),RooFit.LineStyle(kDashed),RooFit.LineColor(kRed))  # Add shape in full ranged dashed
model.plotOn(frame)  # By default only fitted range is shown

# Print fit results 
print "result of fit on all data "
r_full.Print()   
print "result of fit in in signal region (note increased error on signal fraction)" 
r_sig.Print() 

# Draw frame on canvas
c = TCanvas("rf203_ranges","rf203_ranges",600,600) 
gPad.SetLeftMargin(0.15) 
frame.GetYaxis().SetTitleOffset(1.4) 
frame.Draw() 


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

