#!/usr/bin/env python

#
#
# 'VALIDATION AND MC STUDIES' RooFit tutorial macro #801
# 
# A Toy Monte Carlo study that perform cycles of
# event generation and fittting
#
# 
#

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *


from color_palette import *


#### Command line variables ####
batchMode = False


# C r e a t e   m o d e l
# -----------------------

# Declare observable x
x = RooRealVar ("x","x",0,10) 
x.setBins(40) 

# Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their paramaters
mean = RooRealVar("mean","mean of gaussians",5,0,10) 
sigma1 = RooRealVar("sigma1","width of gaussians",0.5) 
sigma2 = RooRealVar("sigma2","width of gaussians",1) 

sig1 = RooGaussian("sig1","Signal component 1",x,mean,sigma1)   
sig2 = RooGaussian("sig2","Signal component 2",x,mean,sigma2)   

# Build Chebychev polynomial p.d.f.  
a0 = RooRealVar("a0","a0",0.5,0.,1.) 
a1 = RooRealVar("a1","a1",-0.2,-1,1.) 
bkg = RooChebychev("bkg","Background",x,RooArgList(a0,a1)) 

# Sum the signal components into a composite signal p.d.f.
sig1frac = RooRealVar("sig1frac","fraction of component 1 in signal",0.8,0.,1.) 
sig = RooAddPdf("sig","Signal",RooArgList(sig1,sig2), RooArgList(sig1frac)) 

# Sum the composite signal and background 
nbkg =  RooRealVar("nbkg","number of background events,",150,0,1000) 
nsig =  RooRealVar("nsig","number of signal events",150,0,1000) 
model = RooAddPdf("model","g1+g2+a",RooArgList(bkg,sig),RooArgList(nbkg,nsig)) 



# C r e a t e   m a n a g e r
# ---------------------------

# Instantiate RooMCStudy manager on model with x as observable and given choice of fit options
#
# The Silence() option kills all messages below the PROGRESS level, leaving only a single message
# per sample executed, and any error message that occur during fitting
#
# The Extended() option has two effects: 
#    1) The extended ML term is included in the likelihood and 
#    2) A poisson fluctuation is introduced on the number of generated events 
#
# The FitOptions() given here are passed to the fitting stage of each toy experiment.
# If Save() is specified, the fit result of each experiment is saved by the manager  
#
# A Binned() option is added in this example to bin the data between generation and fitting
# to speed up the study at the expemse of some precision

#mean.setConstant(kTRUE)

rooargs = RooArgSet(x)
mcstudy = RooMCStudy(model, rooargs, \
        #RooCmdArg(RooFit.Binned(kTRUE)), \
                      RooCmdArg(RooFit.Silence()), \
                      RooCmdArg(RooFit.Extended()),  \
                      RooCmdArg(RooFit.FitOptions(RooFit.Save(kTRUE), RooFit.PrintEvalErrors(0) ) ) ) 


# G e n e r a t e   a n d   f i t   e v e n t s
# ---------------------------------------------

# Generate and fit 1000 samples of Poisson(nExpected) events
mcstudy.generateAndFit(100) 



# E x p l o r e   r e s u l t s   o f   s t u d y 
# ------------------------------------------------

# Make plots of the distributions of mean, the error on mean and the pull of mean
frame1 = mcstudy.plotParam(mean, RooFit.Bins(40)) 
frame2 = mcstudy.plotError(mean, RooFit.Bins(40)) 
frame3 = mcstudy.plotPull(mean, RooFit.Bins(40), RooFit.FitGauss(kTRUE)) 

# Plot distribution of minimized likelihood
frame4 = mcstudy.plotNLL( RooFit.Bins(40)) 

# Make some histograms from the parameter dataset
rllist = RooLinkedList()
rllist.Add(RooFit.Binning(50))
rllist.Add(RooFit.YVar(sig1frac, RooFit.Binning(50)))
#hh_cor_a0_s1f = mcstudy.fitParDataSet().createHistogram("hh", rllist)

rllist = RooLinkedList()
rllist.Add(RooFit.Binning(50))
rllist.Add(RooFit.YVar(a1, RooFit.Binning(50)))
#hh_cor_a0_a1  = mcstudy.fitParDataSet().createHistogram("hh", rllist)


#hh_cor_a0_s1f = mcstudy.fitParDataSet().createHistogram("hh",a1, RooFit.YVar(sig1frac)) 
#hh_cor_a0_a1  = mcstudy.fitParDataSet().createHistogram("hh",a0, RooFit.YVar(a1)) 

# Access some of the saved fit results from individual toys
corrHist000 = mcstudy.fitResult(0).correlationHist("c000") 
corrHist127 = mcstudy.fitResult(27).correlationHist("c27") 
corrHist953 = mcstudy.fitResult(53).correlationHist("c53") 



# Draw all plots on a canvas
gStyle.SetPalette(1) 
gStyle.SetOptStat(0) 
c = TCanvas("rf801_mcstudy","rf801_mcstudy",900,900) 
c.Divide(3,3) 
c.cd(1)  ;frame1.Draw() 
c.cd(2)  ;frame2.Draw() 
c.cd(3)  ;frame3.Draw() 
c.cd(4)  ;frame4.Draw() 
#c.cd(5)  ;hh_cor_a0_s1f.Draw("box") 
#c.cd(6)  ;hh_cor_a0_a1.Draw("box") 
c.cd(7)  ;corrHist000.Draw("colz") 
c.cd(8)  ;corrHist127.Draw("colz") 
c.cd(9)  ;corrHist953.Draw("colz") 

################################################################################
################################################################################
print "Trying my saving"
mcstudy.generate(3,100,kTRUE)

dumd = mcstudy.genData(0)
print dumd
print dumd.numEntries()

# Make RooMCStudy object available on command line after
# macro finishes
gDirectory.Add(mcstudy) 

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

