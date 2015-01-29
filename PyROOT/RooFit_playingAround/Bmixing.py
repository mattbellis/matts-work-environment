#!/usr/bin/env python

################################################################################
#
# 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #309
# 
# Projecting p.d.f and data slices in discrete observables 
#
#
#
# 07/2008 - Wouter Verkerke 
# 
################################################################################
import sys
from ROOT import *

# C r e a t e   B   d e c a y   p d f   w it h   m i x i n g 
# ----------------------------------------------------------

################################################################################
# Common parameters
################################################################################
# Decay time observables
dt = RooRealVar("dt","dt",-20,20)

# Discrete observables mixState (B0tag==B0reco?) and tagFlav (B0tag==B0(bar)?)
mixState = RooCategory("mixState","B0/B0bar mixing state")
tagFlav = RooCategory("tagFlav","Flavour of the tagged B0")

# Define state labels of discrete observables
mixState.defineType("mixed",-1)
mixState.defineType("unmixed",+1)
tagFlav.defineType("B0",+1)
tagFlav.defineType("B0bar",-1)

# Model parameters
dm = RooRealVar("dm","delta m(B)",0.472)
tau = RooRealVar("tau","B0 decay time",1.547,1.0,2.0)
tau.setConstant(False)
################################################################################

########################################
# Dataset 0
########################################
# Parameters that differ by data set.
w_0 = RooRealVar("w_0","Flavor Mistag rate",0.01)
dw_0 = RooRealVar("dw_0","Flavor Mistag rate difference between B0 and B0bar",0.00)

# Build a gaussian resolution model
bias_0 = RooRealVar("bias_0","bias_0",0)
sigma_0 = RooRealVar("sigma_0","sigma_0",0.10)
gm_0 = RooGaussModel("gm_0","gauss model 0",dt,bias_0,sigma_0)

# Construct a decay pdf, smeared with single gaussian resolution model
bmix_gm_0 = RooBMixDecay("bmix_0","decay_0",dt,mixState,tagFlav,tau,dm,w_0,dw_0,gm_0,RooBMixDecay.DoubleSided)

# Generate BMixing data with above set of event errors
data_0 = bmix_gm_0.generate(RooArgSet(dt,tagFlav,mixState),20000)

# Perform fit of bmix0 to data_0 
tau.setVal(1.547)
tau.setConstant(False)
bmix_gm_0.fitTo(data_0)


# P l o t   f u l l   d e c a y   d i s t r i b u t i o n 
# ----------------------------------------------------------

# Create frame, plot data and pdf projection (integrated over tagFlav and mixState)
frames = []
for i in range(0,3):
    title = "Inclusive decay distribution %d" % (i)
    frames.append(dt.frame(RooFit.Title(title)))


argset_proj = RooArgSet()
argset_proj.add(tagFlav)
argset_proj.add(mixState)

argset_0 = RooArgSet(bmix_gm_0)

data_0.plotOn(frames[0])
bmix_gm_0.plotOn(frames[0],RooFit.Components(argset_0),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True))

data_0.plotOn(frames[1],RooFit.Cut("mixState==mixState::mixed"),RooFit.MarkerColor(2))
bmix_gm_0.plotOn(frames[1],RooFit.Components(argset_0),RooFit.Slice(mixState,"mixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True),RooFit.LineColor(7))

data_0.plotOn(frames[2],RooFit.Cut("mixState==mixState::unmixed"),RooFit.MarkerColor(8))
bmix_gm_0.plotOn(frames[2],RooFit.Components(argset_0),RooFit.Slice(mixState,"unmixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True),RooFit.LineColor(7))

c = TCanvas("BMix","BMix",1200,400)
c.SetFillColor(0)
c.Divide(3,1)

for i in range(0,3):
    c.cd(i+1)
    gPad.SetLeftMargin(0.15)
    frames[i].GetYaxis().SetTitleOffset(1.4)
    #gPad.SetLogy()
    frames[i].Draw()
    gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

