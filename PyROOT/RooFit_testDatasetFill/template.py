#!/usr/bin/env python

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
##########################################################
##########################################################
##########################################################
################################################

h = TH1F("h1", "h1", 100, -5, 5)

x = RooRealVar ("x","x",-5,5)

mean  = RooRealVar("mean","#mu of Gaussian", 0.000)
sigma = RooRealVar("sigma","Width of Gaussian", 1.000)
gauss = RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

data = gauss.generate(RooArgSet(x), 1000) # RooDataSet
data.fillHistogram(h, RooArgList(x))

data = gauss.generate(RooArgSet(x), 666) # RooDataSet
data.fillHistogram(h, RooArgList(x))

h.Draw()




################################################
##########################################################
##########################################################
##########################################################
if (not batchMode):
#if (1):
  print "a"
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

