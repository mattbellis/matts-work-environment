#!/usr/bin/env python

###############################################################
# intro7.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro7.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList, RooArgusBG, RooAddPdf
from ROOT import RooAbsPdf, RooFormulaVar, RooCategory, RooRealConstant, RooSuperCategory
from ROOT import RooMappedCategory, RooThresholdCategory, RooTruthModel, RooDecay, RooGaussModel
from ROOT import RooAddModel, RooExtendPdf, RooAbsReal
from ROOT import TCanvas, gStyle, gPad

#### Command line variables ####
batchMode = False

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
##########################################################

# Some global style settings
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

#  extended likelihood use
# Build regular Gaussian PDF
x = RooRealVar("x","x",-10,10) 
mean = RooRealVar("mean","mean of gaussian",-3,-10,10) 
sigma = RooRealVar("sigma","width of gaussian",1,0.1,5) 
gauss = RooGaussian("gauss","gaussian PDF",x,mean,sigma)   

# Make extended PDF based on gauss. n will be the expected number of events
n = RooRealVar("n","number of events",10000,0,20000) 
egauss = RooExtendPdf("egauss","extended gaussian PDF",gauss,n) 

# Generate events from extended PDF
# The default number of events to generate is taken from gauss.expectedEvents()
# but can be overrided using a second argument
data = egauss.generate(RooArgSet(x))  # RooDataSet

# Fit PDF to dataset in extended mode (selected by fit option "e")
egauss.fitTo(data,"mhe") 

# Plot both on a frame 
xframe = x.frame() # RooPlot
data.plotOn(xframe, RooLinkedList()) 
egauss.plotOn(xframe,RooFit.Normalization(1.0,RooAbsReal.RelativeExpected))  # select intrinsic normalization
xframe.Draw()   

# Make an extended gaussian PDF where the number of expected events
# is counted in a limited region of the dependent range
x.setRange("cut",-4,2) 
mean2 = RooRealVar("mean2","mean of gaussian",-3) 
sigma2 = RooRealVar("sigma2","width of gaussian",1) 
gauss2 = RooGaussian("gauss2","gaussian PDF 2",x,mean2,sigma2)   

n2 = RooRealVar("n2","number of events",10000,0,20000) 
egauss2 = RooExtendPdf("egauss2","extended gaussian PDF w limited range",gauss2,n2,"cut") 

egauss2.fitTo(data,"mhe")
print "fitted number of events in data in range (-6,0) = " + str(n2.getVal()) 

# Adding two extended PDFs gives an extended sum PDF

mean.setVal(3.0)   
sigma.setVal(2.0)

# Note that we omit coefficients when adding extended PDFS
sumgauss = RooAddPdf("sumgauss","sum of two extended gauss PDFs",RooArgList(egauss,egauss2)) 
sumgauss.plotOn(xframe, RooFit.Normalization(1.0,RooAbsReal.RelativeExpected), RooFit.LineColor(2))  # select intrinsic normalization
xframe.Draw()   

# Note that in the plot sumgauss does not follow the normalization of the data
# because its expected number was intentionally chosen not to match the number of events in the data

# If no special 'cut normalizations' are needed (as done in egauss2), there is a shorthand 
# way to construct an extended sumpdf:

sumgauss2 = RooAddPdf("sumgauss2","extended sum of two gaussian PDFs",
RooArgList(gauss,gauss2),RooArgList(n,n2)) 
sumgauss2.plotOn(xframe, RooFit.Normalization(1.0, RooAbsReal.RelativeExpected), RooFit.LineColor(3))  # select intrinsic normalization
xframe.Draw()  

gPad.Update()

# Note that sumgauss2 looks different from sumgauss because for gauss2 the expected number
# of event parameter n2 now applies to the entire gauss2 area, whereas in egauss2 it was
# constructed to represent the number of events in the range (-4,-2). If we would use a separate
# parameter n3, set to 10000, to represent the number of events for gauss2 in sumgauss2, then
# sumgauss and sumgauss2 would be indentical.

##########################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

