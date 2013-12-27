#!/usr/bin/env python
################################################################################
#
# 'SPECIAL PDFS' RooFit tutorial macro #707
# 
# Using non-parametric (multi-dimensional) kernel estimation p.d.f.s
#
#
#
# 07/2008 - Wouter Verkerke 
# 
################################################################################

from ROOT import *
from array import *

# C r e a t e   l o w   s t a t s   1 - D   d a t a s e t 
# -------------------------------------------------------

# Create a toy pdf for sampling
x = RooRealVar("x","x",0,20) 
p = RooPolynomial("p","p",x,RooArgList(RooFit.RooConst(0.01),RooFit.RooConst(-0.01),RooFit.RooConst(0.0004))) 

# Sample 500 events from p
data1 = p.generate(RooArgSet(x),200) # RooDataSet



# C r e a t e   1 - D   k e r n e l   e s t i m a t i o n   p d f
# ---------------------------------------------------------------

# Create adaptive kernel estimation pdf. In this configuration the input data
# is mirrored over the boundaries to minimize edge effects in distribution
# that do not fall to zero towards the edges
kest1 = RooNDKeysPdf("kest1","kest1",x,data1,RooKeysPdf.MirrorBoth, 0.25) 
#kest1 = RooNDKeysPdf("kest1","kest1",x,data1, "m") 

# An adaptive kernel estimation pdf on the same data without mirroring option
# for comparison
kest2 = RooNDKeysPdf("kest2","kest2",x,data1,RooKeysPdf.NoMirror) 


# Adaptive kernel estimation pdf with increased bandwidth scale factor
# (promotes smoothness over detail preservation)
kest3 = RooNDKeysPdf("kest1","kest1",x,data1,RooKeysPdf.MirrorBoth,2) 


# Plot kernel estimation pdfs with and without mirroring over data
frame = x.frame(RooFit.Title("Adaptive kernel estimation pdf with and w/o mirroring"),RooFit.Bins(20)) # RooPlot 
data1.plotOn(frame) 
kest1.plotOn(frame) 
kest2.plotOn(frame,RooFit.LineStyle(kDashed),RooFit.LineColor(kRed)) 


# Plot kernel estimation pdfs with regular and increased bandwidth
frame2 = x.frame(RooFit.Title("Adaptive kernel estimation pdf with regular, increased bandwidth")) # RooPlot 
kest1.plotOn(frame2) 
kest3.plotOn(frame2,RooFit.LineColor(kMagenta)) 



c = TCanvas("rf707_kernelestimation","rf707_kernelestimation",1200,800) # TCanvas
c.Divide(1,2) 
c.cd(1) 
gPad.SetLeftMargin(0.15) 
frame.GetYaxis().SetTitleOffset(1.4) 
frame.Draw() 

c.cd(2) 
gPad.SetLeftMargin(0.15) 
frame2.GetYaxis().SetTitleOffset(1.8) 
frame2.Draw() 

        
## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
          rep = rep[0]





