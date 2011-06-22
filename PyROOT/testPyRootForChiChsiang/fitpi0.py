#!/usr/bin/env python

import sys
from optparse import OptionParser
parser = OptionParser()
(options, args) = parser.parse_args()

from ROOT import *
t= TChain("ntp5")
t.Add(args[0])

mass= RooRealVar("mass","mass",0.100,0.160)
aset= RooArgSet(mass,"aset")
data= RooDataSet("data","data",aset)

nentries = t.GetEntries()
for n in xrange (nentries):
    t.GetEntry(n)
    for i in xrange (t.npi0):
        if ( t.pi0Mass[i]>mass.getMin() and t.pi0Mass[i]<mass.getMax() ):
            mass.setVal(t.pi0Mass[i])
            data.add(aset)

data.Print("v")
# create a model
mean= RooRealVar("mean","mean",0.125,0.145)
sigL= RooRealVar("sigL","sigL",0.006,0.001,0.015)
sigR= RooRealVar("sigR","sigR",0.006,0.001,0.015)
peak= RooBifurGauss("peak","peak", mass, mean, sigL, sigR)
a1= RooRealVar("a1","slope",0,-100,100)
bkg= RooPolynomial("bkg","bkg", mass, RooArgList(a1))
frac= RooRealVar("frac","peak fraction",0.03,0,0.3)
model= RooAddPdf("model","model",peak,bkg,frac)

model.fitTo(data, RooFit.Hesse(0))
frame= mass.frame()
data.plotOn(frame)
model.plotOn(frame)
frame.Draw()

del t
