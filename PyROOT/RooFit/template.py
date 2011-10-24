#!/usr/bin/env python

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar, RooCategory, RooRealConstant, RooSuperCategory
from ROOT import RooMappedCategory, RooThresholdCategory, RooTruthModel, RooDecay, RooGaussModel
from ROOT import RooAddModel
from ROOT import TCanvas

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
##########################################################
##########################################################
##########################################################
################################################


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

