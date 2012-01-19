#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys

#############################################################
# Root stuff
#############################################################
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

from array import *

gROOT.Reset()
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)

############################################
# Grab the infile if there is one.
infilename = None
if len(sys.argv)>0 and sys.argv[1] != None:
  infilename = sys.argv[1]
############################################

#psf_lo = 0.900
#psf_hi = 1.020
psf_lo = 0.800
psf_hi = 1.000
x = RooRealVar("x","m_{ES}", 5.2, 5.3)
y = RooRealVar("y","Delta E", -0.2, 0.2)
z = RooRealVar("z","NN", psf_lo, psf_hi)

nevents=0
name = "dataset_%d" % (0)
dataset = RooDataSet(name, name, RooArgSet(x,y,z) )
ds_z = RooDataSet("ds_z", "ds_z", RooArgSet(z) )

max_events = 10000
infile = open(infilename, "r")
for line in infile.readlines():
    dx = float(line.split()[0])
    dy = float(line.split()[1])
    NNval = float(line.split()[2])
    x.setVal(dx)
    y.setVal(dy)
    z.setVal(NNval)

    if nevents<max_events:
        if NNval>psf_lo and NNval<psf_hi:
            # Run this check otherwise the fit won't converge.
            dataset.add(RooArgSet(x,y,z))
            ds_z.add(RooArgSet(z))

            nevents += 1


print "size: %d" % ( ds_z.numEntries() )

name = "rookeyspdf_test.root"
root_out_file = TFile(name, "RECREATE")
w = RooWorkspace("w","my workspace") 





frames = []
for i in range(0,9):
    frames.append(x.frame(RooFit.Bins(200)))
    frames[i].SetMinimum(0)

    frames[i].GetYaxis().SetNdivisions(4)
    frames[i].GetXaxis().SetNdivisions(6)

    frames[i].GetYaxis().SetLabelSize(0.06)
    frames[i].GetXaxis().SetLabelSize(0.06)

    frames[i].GetXaxis().CenterTitle()
    frames[i].GetXaxis().SetTitleSize(0.09)
    frames[i].GetXaxis().SetTitleOffset(1.0)

    frames[i].GetYaxis().CenterTitle()
    frames[i].GetYaxis().SetTitleSize(0.09)
    frames[i].GetYaxis().SetTitleOffset(1.0)

    frames[i].SetMarkerSize(0.01)

################################################################################
################################################################################
#kest1 = RooNDKeysPdf("kest1","kest1",RooArgList(z),dataset,RooNDKeysPdf.MirrorBoth)
#kest2 = RooNDKeysPdf("kest1","kest1",RooArgList(z),dataset,RooNDKeysPdf.NoMirror, 0.5)
kest1 = RooNDKeysPdf("kest1","kest1",z,ds_z,RooNDKeysPdf.MirrorBoth)
kest2 = RooNDKeysPdf("kest2","kest2",z,ds_z,RooNDKeysPdf.NoMirror, 0.5)
kest3 = RooNDKeysPdf("kest3","kest3",z,ds_z,RooNDKeysPdf.NoMirror, 1.0)

z.setBins(200, "cache")
kc = RooCachedPdf("kc","kc",kest2)

print kc

rllist = RooLinkedList()
rllist.Add(RooFit.Binning(200))

# RooDataHist (?)
#rdh = kc.createHistogram("rdh", x, rllist)
rdh = kc.getCacheHist(RooArgSet(z))

print rdh

# RooHistPdf
rhp = RooHistPdf("rhp", "rhp", RooArgSet(z), rdh)





# Try fitting
nsig = RooRealVar ("nsig","# sig events",150)
nsig.setConstant(kFALSE)

#fit_func = RooExtendPdf("fit_func", "Extended function for background", kest2, nsig)
fit_func = RooExtendPdf("fit_func", "Extended function for background", kest1, nsig)

fit_results = fit_func.fitTo(ds_z, 
                             RooFit.Extended(kTRUE), 
                             RooFit.Save(kTRUE), 
                             RooFit.Strategy(2), 
                             RooFit.PrintLevel(-1) ) # RooF

fit_results.Print("v")



# Plot kernel estimation pdfs with and without mirroring over data
frames[0] = z.frame(RooFit.Title("Adaptive kernel estimation pdf with and w/o mirroring"),RooFit.Bins(200)) # RooPlot
dataset.plotOn(frames[0])
kest1.plotOn(frames[0],RooFit.LineColor(kMagenta))
kest2.plotOn(frames[0],RooFit.LineStyle(kDashed),RooFit.LineColor(kRed))
kest3.plotOn(frames[0],RooFit.LineStyle(kDashed),RooFit.LineColor(kGreen))
#kc.plotOn(frames[0],RooFit.LineStyle(kDotted),RooFit.LineColor(kBlue))
rhp.plotOn(frames[0],RooFit.LineStyle(kDashed),RooFit.LineColor(kBlue))



c = TCanvas("rf707_kernelestimation","rf707_kernelestimation",10,10,1200,800) # TCanvas
c.Divide(1,1)
c.cd(1)
gPad.SetLeftMargin(0.15)
frames[0].GetYaxis().SetTitleOffset(1.4)
frames[0].Draw()
gPad.Update()



# Close the root file
getattr(w,'import')(rhp)
getattr(w,'import')(fit_results)
#getattr(w,'import')(kc)
w.Write()
root_out_file.Close()






################################################################################


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]


