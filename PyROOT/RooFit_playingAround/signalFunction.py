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
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

#################################
# Build two PDFs
#################################
x = RooRealVar("x","m_{ES}",5.2,5.3) 
y = RooRealVar("y","#Delta E",-0.3,0.2) 

#################################
# Signal PDF
#################################
meandE = RooRealVar("meandE","mean of gaussian dE", 0.000)
sigmadE = RooRealVar("sigmadE","width of gaussian dE", 0.020)
gaussdE = RooGaussian("gaussdE", "gaussian dE PDF", y, meandE, sigmadE)

meanCB = RooRealVar("mCB","m of gaussian of CB", 5.279)
sigmaCB = RooRealVar("sigmaCB","width of gaussian in CB", 0.0028)
alphaCB = RooRealVar("alphaCB", "alpha of CB", 2.0)
nCB = RooRealVar("nCB","n of CB", 1.0)
cb = RooCBShape("gauss2", "Crystal Barrel Shape PDF", x, meanCB, sigmaCB, alphaCB, nCB)

sigProd = RooProdPdf("sig","gaussdE*cb",RooArgList(gaussdE, cb)) 

#################################
# Background PDF
#################################
# Build polynomial background
p1 = RooRealVar("poly1","1st order coefficient for polynomial",-0.5) 
rarglist = RooArgList(p1)
polyy = RooPolynomial("polyy","Polynomial PDF",y, rarglist);

# Build Argus background PDF
argpar = RooRealVar("argpar","argus shape parameter",-20.0)
cutoff = RooRealVar("cutoff","argus cutoff",5.29)
argus = RooArgusBG("argus","Argus PDF",x,cutoff,argpar)

# Multiply the components
bkgdProd = RooProdPdf("bkgd","argus*polyy",RooArgList(argus,polyy)) 

sigfrac = RooRealVar("sigfrac","fraction of signal", fractionamount)

total = RooAddPdf("total","sig + bkgd",RooArgList(sigProd, bkgdProd), RooArgList(sigfrac))

