#!/usr/bin/env python

from ROOT import *
from array import *

################################################################################
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' RooFit tutorial macro #502
# 
# Creating and writing a workspace
#
#
# 07/2008 - Wouter Verkerke 
#
################################################################################

# C r e a t e   m o d e l   a n d   d a t a s e t
# -----------------------------------------------

# Declare observable x
x = RooRealVar("x","x",0,10)

# Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their paramaters
mean = RooRealVar("mean","mean of gaussians",5,0,10)
sigma1 = RooRealVar("sigma1","width of gaussians",0.5)
sigma2 = RooRealVar("sigma2","width of gaussians",1)

sig1 = RooGaussian("sig1","Signal component 1",x,mean,sigma1)  
sig2 = RooGaussian("sig2","Signal component 2",x,mean,sigma2)  

# Build Chebychev polynomial p.d.f.  
a0 = RooRealVar("a0","a0",0.5,0.,1.)
a1 = RooRealVar("a1","a1",-0.2,0.,1.)
bkg = RooChebychev("bkg","Background",x,RooArgList(a0,a1))

# Sum the signal components into a composite signal p.d.f.
sig1frac = RooRealVar("sig1frac","fraction of component 1 in signal",0.8,0.,1.)
sig = RooAddPdf("sig","Signal",RooArgList(sig1,sig2),RooArgList(sig1frac))

# Sum the composite signal and background 
bkgfrac = RooRealVar("bkgfrac","fraction of background",0.5,0.,1.)
model = RooAddPdf("model","g1+g2+a",RooArgList(bkg,sig),RooArgList(bkgfrac))

# Generate a data sample of 1000 events in x from model
data = model.generate(RooArgSet(x),1000)



# C r e a t e   w o r k s p a c e ,   i m p o r t   d a t a   a n d   m o d e l
# -----------------------------------------------------------------------------

# Create a new empty workspace
w = RooWorkspace("w","workspace")

# Import model and all its components into the workspace
getattr(w,'import')(model) 


# Import data into the workspace
getattr(w,'import')(data)

# Print workspace contents
w.Print()



# S a v e   w o r k s p a c e   i n   f i l e
# -------------------------------------------

# Save the workspace into a ROOT file
w.writeToFile("rf502_workspace.root")


# Workspace will remain in memory after macro finishes
gDirectory.Add(w)


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]
