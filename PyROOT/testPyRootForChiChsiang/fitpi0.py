#!/usr/bin/env python

import sys
from optparse import OptionParser
parser = OptionParser()
(options, args) = parser.parse_args()

import ROOT 
t= ROOT.TChain("ntp5")
t.Add(args[0])

mass= ROOT.RooRealVar("mass","mass",0.100,0.160)
aset= ROOT.RooArgSet(mass,"aset")
data= ROOT.RooDataSet("data","data",aset)

nentries = t.GetEntries()
for n in xrange (nentries):
    t.GetEntry(n)
    for i in xrange (t.npi0):
        if ( t.pi0Mass[i]>mass.getMin() and t.pi0Mass[i]<mass.getMax() ):
            mass.setVal(t.pi0Mass[i])
            data.add(aset)

data.Print("v")
# create a model
mean= ROOT.RooRealVar("mean","mean",0.125,0.145)
sigL= ROOT.RooRealVar("sigL","sigL",0.006,0.001,0.015)
sigR= ROOT.RooRealVar("sigR","sigR",0.006,0.001,0.015)
peak= ROOT.RooBifurGauss("peak","peak", mass, mean, sigL, sigR)
a1= ROOT.RooRealVar("a1","slope",0,-100,100)
bkg= ROOT.RooPolynomial("bkg","bkg", mass, ROOT.RooArgList(a1))
frac= ROOT.RooRealVar("frac","peak fraction",0.03,0,0.3)
model= ROOT.RooAddPdf("model","model",peak,bkg,frac)

model.fitTo(data, ROOT.RooFit.Hesse(0))
frame= mass.frame()
data.plotOn(frame)
model.plotOn(frame)
frame.Draw()

del t
