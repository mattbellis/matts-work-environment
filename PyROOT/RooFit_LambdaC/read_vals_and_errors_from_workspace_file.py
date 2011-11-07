#!/usr/bin/env python

from ROOT import *

from optparse import OptionParser

from array import *
import sys

gROOT.Reset()
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)

################################################################################
parser = OptionParser()
parser.add_option("--var", dest="var", default="branching_fraction", \
        help="Fit variable for which to dump values.")
parser.add_option("-v", dest="verbose", action="count", \
        help="Set the level of verbosity.")

(options, args) = parser.parse_args()

workspace_file = args[0]

################################################################################
# R e a d   w o r k s p a c e   f r o m   f i l e
# -----------------------------------------------
# Open input file with workspace 
f = TFile (workspace_file)

# Retrieve workspace from file
in_wname = workspace_file.split('/')[-1].split('.root')[0]
print in_wname
w = f.Get(in_wname) # RooWorkspace

if options.verbose>0:
    w.Print()
elif options.verbose>1:
    w.Print("v")

all_vars = w.allVars()
if options.verbose>0:
    all_vars.Print("v")

var = w.var(options.var)
value = var.getVal()
err_hi = var.getErrorHi()
err_lo = var.getErrorLo()

print "\n%s: %f\thi err:%f\t\tlo err:%f\n" % (options.var, value, err_hi, err_lo)

