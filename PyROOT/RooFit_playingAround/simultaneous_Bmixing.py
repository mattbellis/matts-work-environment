#!/usr/bin/env python

################################################################################
#
# Based on 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #309
# 
# Projecting p.d.f and data slices in discrete observables 
#
# 05/2011 - M. Bellis
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
dm = RooRealVar("dm","delta m(B)",0.472,0.0,1.5)
tau = RooRealVar("tau","B0 decay time",1.547,1.0,2.0)
tau.setConstant(False)
################################################################################

########################################
# Dataset 0
########################################
# Parameters that differ by data set.
w_0 = RooRealVar("w_0","Flavor Mistag rate",0.01,0.0,1.0)
dw_0 = RooRealVar("dw_0","Flavor Mistag rate difference between B0 and B0bar",0.00,0.0,1.0)

# Build a gaussian resolution model
bias_0 = RooRealVar("bias_0","bias_0",0,-0.1,0.1)
sigma_0 = RooRealVar("sigma_0","sigma_0",0.04,0.0,1.0)
gm_0 = RooGaussModel("gm_0","gauss model 0",dt,bias_0,sigma_0)

# Construct a decay pdf, smeared with single gaussian resolution model
bmix_gm_0 = RooBMixDecay("bmix_0","decay_0",dt,mixState,tagFlav,tau,dm,w_0,dw_0,gm_0,RooBMixDecay.DoubleSided)

########################################
# Dataset 1
########################################
# Parameters that differ by data set.
w_1 = RooRealVar("w_1","Flavor Mistag rate",0.01,0.0,1.0)
dw_1 = RooRealVar("dw_1","Flavor Mistag rate difference between B0 and B0bar",0.00,0.0,1.0)

# Build a gaussian resolution model
bias_1 = RooRealVar("bias_1","bias_1",0,-0.1,0.1)
sigma_1 = RooRealVar("sigma_1","sigma_1",0.20,0.0,1.0)
gm_1 = RooGaussModel("gm_1","gauss model 1",dt,bias_1,sigma_1)

# Construct a decay pdf, smeared with single gaussian resolution model
bmix_gm_1 = RooBMixDecay("bmix_1","decay_1",dt,mixState,tagFlav,tau,dm,w_1,dw_1,gm_1,RooBMixDecay.DoubleSided)


################################################################################
# Define category to distinguish ee and mm samples events
################################################################################
sample = RooCategory("sample","sample")
sample.defineType("ee")
sample.defineType("mm")

# Generate BMixing data with above set of event errors
# for both datasets.
data_0 = bmix_gm_0.generate(RooArgSet(dt,tagFlav,mixState),20000)
data_1 = bmix_gm_1.generate(RooArgSet(dt,tagFlav,mixState),10000)

# Construct combined dataset in (x,sample)
combData = RooDataSet("combData","combined data",RooArgSet(dt,tagFlav,mixState),
        RooFit.Index(sample),
        RooFit.Import("ee",data_0),
        RooFit.Import("mm",data_1)) 

# Construct a simultaneous pdf using category sample as index
simPdf = RooSimultaneous("simPdf","simultaneous pdf",sample)

# Associate bmix0 with the ee state and bmix1 with the mm state
simPdf.addPdf(bmix_gm_0,"ee")
simPdf.addPdf(bmix_gm_1,"mm")

# Perform simultaneous fit of bmix0 to data_0 and bmix1 to data_1
tau.setVal(1.547)
tau.setConstant(False)
simPdf.fitTo(combData) ;


# P l o t   f u l l   d e c a y   d i s t r i b u t i o n 
# ----------------------------------------------------------

# Create frame, plot data and pdf projection (integrated over tagFlav and mixState)
frames = []
for i in range(0,9):
    title = "Inclusive decay distribution %d" % (i)
    frames.append(dt.frame(RooFit.Title(title)))


# Plot both datasets and the combined PDF
combData.plotOn(frames[0])
simPdf.plotOn(frames[0])

# Plot the two data samples separately.
combData.plotOn(frames[1],RooFit.Cut("sample==sample::ee"),RooFit.MarkerColor(2))
combData.plotOn(frames[2],RooFit.Cut("sample==sample::mm"),RooFit.MarkerColor(8))

argset_proj = RooArgSet()
argset_proj.add(tagFlav)
argset_proj.add(mixState)

argset_0 = RooArgSet(bmix_gm_0)
argset_1 = RooArgSet(bmix_gm_1)

# Plot data sample 0 and the resultant fit.
data_0.plotOn(frames[3])
bmix_gm_0.plotOn(frames[3],RooFit.Components(argset_0),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True))

data_0.plotOn(frames[4],RooFit.Cut("mixState==mixState::mixed"),RooFit.MarkerColor(2))
bmix_gm_0.plotOn(frames[4],RooFit.Components(argset_0),RooFit.Slice(mixState,"mixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True),RooFit.LineColor(7))

data_0.plotOn(frames[5],RooFit.Cut("mixState==mixState::unmixed"),RooFit.MarkerColor(8))
bmix_gm_0.plotOn(frames[5],RooFit.Components(argset_0),RooFit.Slice(mixState,"unmixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_0,True),RooFit.LineColor(7))

# Plot data sample 1 and the resultant fit.
data_1.plotOn(frames[6])
bmix_gm_1.plotOn(frames[6])

data_1.plotOn(frames[7],RooFit.Cut("mixState==mixState::mixed"),RooFit.MarkerColor(2))
bmix_gm_1.plotOn(frames[7],RooFit.Components(argset_1),RooFit.Slice(mixState,"mixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_1,True),RooFit.LineColor(7))

data_1.plotOn(frames[8],RooFit.Cut("mixState==mixState::unmixed"),RooFit.MarkerColor(8))
bmix_gm_1.plotOn(frames[8],RooFit.Components(argset_1),RooFit.Slice(mixState,"unmixed"),RooFit.ProjWData(RooArgSet(argset_proj),data_1,True),RooFit.LineColor(7))

c = TCanvas("simultaneous_BMix","simultaneous BMix",1200,1000)
c.SetFillColor(0)
c.Divide(3,3)

for i in range(0,9):
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

