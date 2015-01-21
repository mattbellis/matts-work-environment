#!/usr/bin/env python

###############################################################
###############################################################

import os
import sys
from optparse import OptionParser

#### Command line variables ####
parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("-b", "--num-bkg-events", dest="num_bkg", default=1000, help="Number of background events in fit")
parser.add_option("-s", "--num-sig-events", dest="num_sig", default=100, help="Number of signal events in fit")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--fixed-number", dest="fixed_num", action="store_true", default=False, help="Use a fixed number for the signal; \
        do NOT sample from a Poisson distribution.")

(options, args) = parser.parse_args()

# Read in the starting values
starting_values = []

################################################
from ROOT import *
gSystem.Load('libRooFit')
################################################
################################################################################

from pdf_definitions import *

##################################
# Fit dimensionality
##################################
dim = int(options.dimensionality)

xlo = 0.0; xhi = 10.0
ylo = 0.0; yhi = 10.0
zlo = 0.0; zhi = 10.0
data_ranges = [[xlo,xhi], [ylo,yhi], [zlo,zhi]]

# RooRealVars (axis)
x,y,z = build_xyz(data_ranges)

################################################################################
################################################################################

sub_funcs = []

# Grab the fit functions and everything else which is needed.
my_pars, sub_funcs_list, fit_func = tot_PDF([x,y,z])

# Create a dictionary of the pars
pars_dict = {}
for p in my_pars:
    print p
    pars_dict[p.GetName()] = p

##############################################
# Set num sig/bkg by hand
##############################################
pars_dict["nbkg"].setVal( float(options.num_bkg) )
pars_dict["nsig"].setVal( float(options.num_sig) )

pars_dict["c_x_bkg"].setVal(-0.5)
pars_dict["c_y_bkg"].setVal(-0.5)
pars_dict["c_z_bkg"].setVal(-0.5)

pars_dict["mean_x_sig"].setVal(5.0)
pars_dict["sigma_x_sig"].setVal(0.5)

pars_dict["mean_y_sig"].setVal(5.0)
pars_dict["sigma_y_sig"].setVal(0.5)

pars_dict["mean_z_sig"].setVal(5.0)
pars_dict["sigma_z_sig"].setVal(0.5)

fixed_tag = ""
fixed_bool = kTRUE
if options.fixed_num:
    fixed_tag = "_fixedSig"
    fixed_bool = kFALSE

######################################
# Toy fits
######################################
# Create study manager for binned likelihood fits of a Gaussian pdf in 10 bins
rooargs = RooArgSet(x,y,z)

# Make sure I *don't* have the Binned option or things die.
mcstudy = RooMCStudy(fit_func, rooargs, \
    RooCmdArg(RooFit.Extended(fixed_bool)), \
    RooCmdArg(RooFit.Silence()), \
    RooCmdArg(RooFit.FitOptions(RooFit.Save(kTRUE))))


dirname = "mcstudies"
filename = "%s/mcstudies_bkg%d_sig%d%s_%s.dat" % ( dirname, int(options.num_bkg), int(options.num_sig), fixed_tag, "%04d")

mcstudy.generate(1, int(options.num_bkg)+int(options.num_sig), kTRUE, filename )
