#!/usr/bin/env python

###############################################################
# Matt Bellis
# bellis@slac.stanford.edu
###############################################################

import sys
from optparse import OptionParser

#### Command line variables ####

parser = OptionParser()
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")
parser.add_option("--batch", dest="batch", default=False, action="store_true", help="Run in batch mode.")

(options, args) = parser.parse_args()

from ROOT import *

#######################################################
gROOT.Reset()
################################################

# Define the x variable.
x = RooRealVar("x","x",-10,10)

# Build signal PDF
# Gaussian PDF
sig_mean = RooRealVar("sig_mean","Mean of gaussian",-1)
sig_sigma = RooRealVar("sig_sigma","Width of gaussian",0.5)
sig_gauss = RooGaussian("sig_gauss","Gaussian PDF",x,sig_mean,sig_sigma)

# Build background PDF
# Linear PDF
bkg_p1 = RooRealVar("bkg_p1","Linear coefficient",-0.1)
rarglist = RooArgList(bkg_p1)
bkg_linear = RooPolynomial("bkg_linear","Polynomial PDF",x,rarglist);

# Build total PDF
sig_fraction = RooRealVar("sig_fraction","Fraction of total that is signal",0.2)
total = RooAddPdf("total","sig_gauss + bkg_linear",RooArgList(sig_gauss, bkg_linear), RooArgList(sig_fraction))


# Generate a toy MC set
data = total.generate(RooArgSet(x), 10000) # RooDataSet

xframe = x.frame()
data.plotOn(xframe,RooLinkedList())
total.plotOn(xframe)
xframe.Draw()



## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


