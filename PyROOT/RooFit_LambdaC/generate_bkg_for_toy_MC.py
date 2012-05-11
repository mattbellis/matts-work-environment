#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import os
import sys
from optparse import OptionParser

#### Command line variables ####
parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-b", "--num-bkg-events", dest="num_bkg", default=600, help="Number of background events in fit")
parser.add_option("-s", "--num-sig-events", dest="num_sig", default=60, help="Number of signal events in fit")
parser.add_option("-N", "--num-studies", dest="num_studies", default=10, help="Number of toy studies to run")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--fixed-number", dest="fixed_num", action="store_true", default=False, help="Use a fixed number for the signal; \
        do NOT sample from a Poisson distribution.")
parser.add_option("--ntp", dest="ntp", default="ntp1", help="Baryon [LambdaC, Lambda0]")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")
parser.add_option("--use-single-cb", dest="use_single_cb", action = "store_true", default = False, help="Use the single CB in Delta E")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")
parser.add_option("--starting-vals-file", dest="starting_vals_file", default="default_starting_vals_file.txt", help="File to use for the starting values")
parser.add_option("--workspace", dest="workspace_file", help="File from which to grab the workspace and NN vals.")


(options, args) = parser.parse_args()

# Read in the starting values
starting_values = []
if options.starting_vals_file != None:
  infile = open(options.starting_vals_file)
  for line in infile:
    if line[0] != "#":
      start_val = line.split()
      starting_values.append([start_val[0], start_val[2], int(start_val[3]) ] )

#print starting_values

use_double_cb = True
if options.use_single_cb:
    use_double_cb = False

################################################
from ROOT import *
gSystem.Load('libRooFit')
################################################
##################################
# Fit deimensionality
##################################
dim = int(options.dimensionality)

################################################################################
# Must call this first to set the ranges for the parametric step function
################################################################################
#from nn_limits_and_binning import *

#psf_lo, psf_hi, vary_limits = 0.0, 0.0, 0.0
#psf_lo, psf_hi, vary_limits = nn_fit_params(options.baryon, options.ntp, 0)

################################################################################
################################################################################

from pdf_definitions import *
from my_roofit_utilities import *
from file_map import *

pass_info = fit_pass_info(options.baryon, options.ntp, int(options.my_pass))
mes_lo = pass_info[4][0]
mes_hi = pass_info[4][1]

deltae_lo = pass_info[5][0]
deltae_hi = pass_info[5][1]

nn_lo = pass_info[6][0]
nn_hi = pass_info[6][1]

data_ranges = [[mes_lo,mes_hi], [deltae_lo,deltae_hi], [nn_lo,nn_hi]]


# RooRealVars (axis)
#x,y,z = build_xyz(psf_lo, psf_hi)
x,y,z = build_xyz(data_ranges)

z.setBins(200, "cache")

# RooDataSets
data = RooDataSet("data","data",RooArgSet(x,y,z))
# We need this one for when we are *only* fitting the NN output.
data_z = RooDataSet("data_z","data_z",RooArgSet(z))

'''
data = RooDataSet("data","data",RooArgSet(x,y,z)) 
if dim==2:
  data = RooDataSet("data","data",RooArgSet(x,y)) 
'''

########################################
# Try setting the starting parameters
########################################
#lo = psf_lo
#hi = psf_hi-0.00001
# Set the starting values from the file
# Must initialize the bin parameters before I return the pars.
#rs_dum = myRooParSF("sig", lo, hi)
#rb_dum = myRooParSF("bkg", lo, hi)
#mypars = get_mypars()


'''
print "Setting starting vals....."
for s in starting_values:
  for p in mypars:
    if s[0] == p.GetName():
      print "%s %f %f" % (p.GetName(), float(s[1]), p.getMax())
      p.setVal(float(s[1]))
      p.setConstant(s[2])
'''


#################################################
# To generate from an extended distribution
#################################################

################################################################################
# This is for when we use the RooParametricStepFunction
# Just make the bins (bh)
#bin_heights, bh = build_the_bins_for_psf(psf_lo, psf_hi, vary_limits)
################################################################################

mypars = []
sub_funcs = []

# Check if a workspace file has been passed in.
# From this we would grab the RooWorkspace object and grab the NN RooHistPdf
# object for the resulting fit.
workspace = None
print "dim: %d" % (dim) 
if options.workspace_file!=None and dim==3:
    wfile = TFile(options.workspace_file, "READ")
    # Workspace object is same name as file, minus the '.root'
    in_wname = options.workspace_file.split('/')[-1].split('.root')[0]
    workspace = wfile.Get(in_wname)
    if workspace==None:
        print "NO WORKSPACE FOUND IN FILE!!!!!!"
        exit(-1)
    print workspace


# Grab the fit functions and everything else which is needed.
dum_pars, sub_funcs_list, fit_func = tot_PDF(x,y,z, data_z,
                  dim, use_double_cb, workspace)
mypars += dum_pars
'''
dum_pars, sub_funcs_list, fit_func = tot_PDF(x,y,z, data_z, bh, vary_limits,
                                 dim, use_double_cb , psf_lo, psf_hi, workspace)
mypars += dum_pars
'''
#print mypars

# Create a dictionary of the pars
# We'll use this later if we need to print stuff out.
#print mypars
pars_d = {}
for p in mypars:
    pars_d[p.GetName()] = p

#rpsf_s, rpsf_b, sig_pdf, bkg_pdf, fit_func = tot_PDF( dim, True , lo, hi)


set_starting_values(mypars, starting_values)

##############################################
# Set num sig/bkg by hand
##############################################
pars_d["nbkg"].setVal( float(options.num_bkg) )
#pars_d["nsig"].setVal( float(options.num_sig) )
# Set nsig based on branching fraction
#conv_factor_fit*branching_fraction
my_nsig = float(options.num_sig) 
my_conv_factor_calc = pars_d["conv_factor_calc"].getVal()
my_branching_fraction = my_nsig/my_conv_factor_calc
pars_d["branching_fraction"].setVal(my_branching_fraction)
# Set the fit conv to the calc conv
pars_d["conv_factor_fit"].setVal(my_conv_factor_calc)


"""
x.setRange("FULL",5.2, 5.30)
y.setRange("FULL",-0.2, 0.2)
z.setRange("FULL", psf_lo, psf_hi-0.1)
"""

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


dirname = options.tag
if not os.path.exists (dirname):
  os.makedirs (dirname)


filename = "%s/mcstudies_bkg%d_sig%d%s_%s.dat" % ( dirname, int(options.num_bkg), int(options.num_sig), fixed_tag, "%04d")

mcstudy.generate(int(options.num_studies), int(options.num_bkg)+int(options.num_sig), kTRUE, filename )
