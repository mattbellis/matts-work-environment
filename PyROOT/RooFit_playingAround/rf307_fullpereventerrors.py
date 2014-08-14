#!/usr/bin/env python

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

################################################################################
# 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #307
# 
# Complete example with use of full p.d.f. with per-event errors
#
# 07/2008 - Wouter Verkerke 
################################################################################

# B - p h y s i c s   p d f   w i t h   p e r - e v e n t  G a u s s i a n   r e s o l u t i o n
# ----------------------------------------------------------------------------------------------

# Observables
dt = RooRealVar("dt","dt",-10,10)
dterr = RooRealVar("dterr","per-event error on dt",0.01,10)

# Build a gaussian resolution model scaled by the per-event error = gauss(dt,bias,sigma*dterr)
bias = RooRealVar("bias","bias",0,-10,10)
sigma = RooRealVar("sigma","per-event error scale factor",1,0.1,10)
gm = RooGaussModel("gm1","gauss model scaled bt per-event error",dt,bias,sigma,dterr)

# Construct decay(dt) (x) gauss1(dt|dterr)
tau = RooRealVar("tau","tau",1.548)
decay_gm = RooDecay("decay_gm","decay",dt,tau,gm,RooDecay.DoubleSided)

# C o n s t r u c t   e m p i r i c a l   p d f   f o r   p e r - e v e n t   e r r o r
# -----------------------------------------------------------------

# Use landau p.d.f to get empirical distribution with long tail
pdfDtErr = RooLandau("pdfDtErr","pdfDtErr",dterr,RooFit.RooConst(1),RooFit.RooConst(0.25))
expDataDterr = pdfDtErr.generate(RooArgSet(dterr),10000) # RooDataSet

# Construct a histogram pdf to describe the shape of the dtErr distribution
expHistDterr = expDataDterr.binnedClone() # RooDataHist
pdfErr = RooHistPdf("pdfErr","pdfErr",RooArgSet(dterr),expHistDterr)


# C o n s t r u c t   c o n d i t i o n a l   p r o d u c t   d e c a y _ d m ( d t | d t e r r ) * p d f ( d t e r r )
# ----------------------------------------------------------------------------------------------------------------------

# Construct production of conditional decay_dm(dt|dterr) with empirical pdfErr(dterr)
model = RooProdPdf("model","model",RooArgSet(pdfErr),RooFit.Conditional(RooArgSet(decay_gm),RooArgSet(dt)))

# (Alternatively you could also use the landau shape pdfDtErr)
#RooProdPdf model("model","model",pdfDtErr,RooFit.Conditional(decay_gm,dt))



# S a m p l e,   f i t   a n d   p l o t   p r o d u c t   m o d e l 
# ------------------------------------------------------------------

# Specify external dataset with dterr values to use model_dm as conditional p.d.f.
data = model.generate(RooArgSet(dt,dterr),10000) # RooDataSet



# F i t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
# ---------------------------------------------------------------------

# Specify dterr as conditional observable
model.fitTo(data)



# P l o t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
# ---------------------------------------------------------------------


# Make two-dimensional plot of conditional p.d.f in (dt,dterr)
hh_model = model.createHistogram("hh_model",dt,RooFit.Binning(50),RooFit.YVar(dterr,RooFit.Binning(50))) # TH1
hh_model.SetLineColor(kBlue)


# Make projection of data an dt
frame = dt.frame(RooFit.Title("Projection of model(dt|dterr) on dt")) # RooPlot
data.plotOn(frame)
model.plotOn(frame)

# Make projection of data an dterr
frame1 = dterr.frame(RooFit.Title("Projection of model(dt|dterr) on dterr")) # RooPlot
data.plotOn(frame1)
model.plotOn(frame1)


# Draw all frames on canvas
c = TCanvas("rf307_fullpereventerrors","rf307_fullperventerrors",1200, 400) # TCanvas
c.Divide(3,1)
c.cd(1)
gPad.SetLeftMargin(0.20)
hh_model.GetZaxis().SetTitleOffset(2.5)
hh_model.Draw("surf")

c.cd(2)
gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.6)
frame.Draw()

c.cd(3)
gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.6)
frame1.Draw()




## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

