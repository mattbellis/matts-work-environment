#!/usr/bin/env python

###############################################################
# intro9.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro9.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

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

#  dependents vs parameters
# This example demonstrates RFCs dynamic concept of dependents vs parameters
#
# Definition:
#   'dependent' = a variable of a function that is loaded from a dataset
#   'parameter' = any variable that is not a dependent
#
# For probability density functions this distinction is non-trivial because,
# by construction, they must be normalized over all dependent variables:
#      
#     Int(dDep) Pdf(Dep,Par) == 1    (Dep = set of dependent variables , 
#                                     Par = set parameter variables)
#
# In RooFit, the dep/par status of each variable of a RooAbsPdf object is
# explicitly dynamic: each variable can be a parameter or a dependent
# at any time. This only depends on the context in which the PDF is used, i.e.
# the status of each variables is only explicit when a PDF is associated
# with a dataset, which determines the set of dependents. 
#
# A consequence of this dynamic concept is that a complete set
# of values for all PDF variables does not yield a unique return value:
# the dep/par status of each variables must be specified in addition.
#
# To achieve this dynamic property, the normalization of a PDF is decoupled 
# from the calculation of its raw unnormalized value.
# 

# A simple gaussian PDF has 3 variables: x,mean,sigma
x = RooRealVar("x","x",-10,10) 
mean = RooRealVar("mean","mean of gaussian",-1,-10,10) 
sigma = RooRealVar("sigma","width of gaussian",3,1,20) 
gauss = RooGaussian("gauss","gaussian PDF",x,mean,sigma)   

# For getVal() without any arguments all variables are interpreted as parameters,
# and no normalization is enforced
x.setVal(0)
rawVal = gauss.getVal()  # = exp(-[(x-mean)/sigma]^2)
print "gauss(x=0,mean=-1,width=3)_raw = " + str(rawVal)

# If we supply getVal() with the subset of its variables that should be interpreted as dependents, 
# it will apply the correct normalization for that set of dependents
xnormVal = gauss.getVal(RooArgSet(x))  # NB: RooAbsArg& implicitly converts to RooArgSet(RooAbsArg&) 
print "gauss(x=0,mean=-1,width=3)_normalized_x[-10,10] = " + str(xnormVal)

# gauss.getVal(x) = gauss.getVal() / Int(-10,10) gauss() dx

# If we adjust the limits on x, the normalization will change accordingly
x.setRange(-1,1) 
xnorm2Val = gauss.getVal(RooArgSet(x)) 
print "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1] = " + str(xnorm2Val)

# gauss.getVal(x) = gauss.getVal() / Int(-1,1) gauss() dx

# We can also add sigma as dependent
xsnormVal = gauss.getVal(RooArgSet(x,sigma)) 
print "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1]_width[1,20] = " + str(xsnormVal) + "\n"

# gauss.getVal(RooArgSet(x,sigma)) = gauss.getVal() / Int(-1,1)(1,20) gauss() dx dsigma

# (!) The set of arguments passed in RooAbsPdf::getVal() determines _only_ which
#     the variables are interpreted as dependents/parameters. No values are passed,
#     these are _always_ taken from the arguments supplied in the constructor
#
#     Objects passed in RooAbsPdf::getVal() that are not variables of the PDF are
#     silently ignored

##########################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

