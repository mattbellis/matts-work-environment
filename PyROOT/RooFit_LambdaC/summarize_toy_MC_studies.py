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
from optparse import OptionParser

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--ntp", dest="ntp", default="ntp1", help="Baryon [LambdaC, Lambda0]")
parser.add_option("-m", "--max", dest="max", help="Maximum events to read in")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--results-file", dest="results_filename", default="default_results.root", help="File from which to read the results.")
parser.add_option("--no-gc", dest="no_gc", action = "store_true", default=False, help="Don't use the gaussian constraint")
parser.add_option("--num-sig", dest="num_sig", help="Number of embeded signal events")
parser.add_option("--num-bkg", dest="num_bkg", default=650, help="Number of background events")
parser.add_option("--fixed-num", dest="fixed_num", action="store_true", default=False, \
        help="Use a fixed number of both background and signal.")
parser.add_option("--num-bins", dest="num_bins", default=50, help="Number of bins to use")
parser.add_option("--dir", dest="dir", help="Directory from which to read the pure and embedded study files.")
#parser.add_option("-n", "--num-fits", dest="num_fits", default=10, help="Number of toy studies in the file")
parser.add_option("--pure", dest="pure", action="store_true", default=False, help="Do pure toy MC studies.")
parser.add_option("--embed", dest="embed", action="store_true", default=False, help="Do embedded toy MC studies.")
parser.add_option("--use-double-cb", dest="use_double_cb", action = "store_true", default = False, help="Use the double CB in Delta E")
parser.add_option("--starting-vals-file", dest="starting_vals_file", default="default_starting_vals_file.txt", help="File to use for the starting values")
parser.add_option("--fit-only-sig", dest="fit_only_sig", action = "store_true", default = False, help="Fit only to the signal.")
parser.add_option("--fit-only-bkg", dest="fit_only_bkg", action = "store_true", default = False, help="Fit only to the background.")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")

(options, args) = parser.parse_args()

import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

#from backgroundAndSignal_NEW_def import *
#import backgroundAndSignal_NEW_def 

from pdf_definitions import *
from my_roofit_utilities import *

#######################################################
gROOT.Reset()
gStyle.SetOptStat(0)
gStyle.SetOptFit(0)
#gStyle.SetOptStat(11111)
#gStyle.SetOptFit(111111)
#gStyle.SetStatH(0.6)
#gStyle.SetStatW(0.5)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
#set_palette("palette",100)
# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)
#set_palette("palette",100)

#filename = args[0]

dim = int(options.dimensionality)

num_bins = 25
if options.num_bins:
  num_bins = int(options.num_bins)


max_events = 1e9
if options.max:
    max_events = float(options.max)



################################################
################################################
################################################
################################################################################
# Must call this first to set the ranges for the parametric step function
################################################################################
from nn_limits_and_binning import *

psf_lo, psf_hi, vary_limits = 0.0, 0.0, 0.0
psf_lo, psf_hi, vary_limits = nn_fit_params(options.baryon, options.ntp, 0)

# Redefine psf_lo and psf_hi
from file_map import *

pass_info = fit_pass_info(options.baryon, options.ntp, int(options.my_pass))

print options.baryon
print options.ntp
print options.my_pass
print "THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"
print pass_info

mes_lo = pass_info[4][0]
mes_hi = pass_info[4][1]

deltae_lo = pass_info[5][0]
deltae_hi = pass_info[5][1]

psf_lo = pass_info[6][0]
psf_hi = pass_info[6][1]

print "PSF's"
print "%f %f" % (psf_lo,psf_hi)
data_ranges = [[mes_lo,mes_hi], [deltae_lo,deltae_hi], [psf_lo,psf_hi]]
print "IN SUMRRRRRRRRRRRRRRRRRRRRHHHHHHHHIIIIIIIIZZZZZZZZZEEEEEEEEEE"
print data_ranges

################################################################################


########################################
########################################

rootfile = TFile(options.results_filename, "READ")
#rootfile.ls()
in_wname = options.results_filename.split('/')[-1].split('.root')[0]
print "Root workspace file: %s" % (in_wname)
w = rootfile.Get(in_wname) # RooWorkspace

print "Grabbed the workspace file!"

#w.Print("v")
x = w.var("x")
y = w.var("y")
z = w.var("z")
z.setBins(200, "cache")

total_name = "total"
if options.fit_only_sig:
    total_name = "sig_pdf"
elif options.fit_only_bkg:
    total_name = "bkg_pdf"

################################################################################
'''
# Dump the vars
all_vars = w.allVars()
iter = all_vars.createIterator()
# Build a dictionary
my_vars = {}
nvars = all_vars.getSize()
# Grab the pdf we want to plot
for i in range(0,nvars):
    p = iter.Next()
    print "%s %5.5f" % (p.GetName(), p.getVal())
    my_vars[p.GetName()] = p
'''

################################################################################
all_pdfs = w.allPdfs()
iter = all_pdfs.createIterator()
# Build a dictionary
pdfs = {}
npdfs = all_pdfs.getSize()
# Grab the pdf we want to plot
for i in range(0,npdfs):
    p = iter.Next()
    pdfs[p.GetName()] = p

print "Grabbed all the PDF's......"




data = w.data("dataset_0")
data_z = w.data("dataset_z_0")


keys = rootfile.GetListOfKeys()
nfits = rootfile.GetNkeys()

upper_limits = []
std_devs = []
branching_fractions = []
# Count how many keys are actually fitresult objects
nfits = 0
for k in keys:
    #print k.GetName()
    if k.GetName().find("fitresult")>=0:
        nfits += 1

print "Num fits: %d" % ( nfits )

frames = x.frame(RooFit.Bins(int(options.num_bins)))

# Open the files and read them in for the datasets
#for n in range(0, 1):
#for n in range(0, 50):
for n in range(0, int(nfits)):

    fixed_tag = ""
    if options.fixed_num:
        fixed_tag = "_fixedSig"
    if options.pure:
        infilename = "%s/mcstudies_bkg%d_sig%d%s_%04d.dat" % \
            (options.dir, int(options.num_bkg),int(options.num_sig),fixed_tag,n)
    elif options.embed:
        infilename = "%s/mcstudies_bkg%d_embed_sig%d%s_%04d.dat" % \
            (options.dir, int(options.num_bkg),int(options.num_sig),fixed_tag,n)

    print "Opening %s ............................." % (infilename)
    infile = open(infilename)

    '''
    del data_z
    data_z, data = read_file_return_dataset(infile, x,y,z,\
                       data_ranges, int(options.dimensionality), max_events, n)
    '''
    #getattr(w,'import')(data)
    del data
    name = "dataset_%d" % (n)
    data = w.data(name)


    print "Num data points: %d" % ( data.numEntries() )

    ################################################################################
    # Dump the vars
    all_vars = w.allVars()
    iter = all_vars.createIterator()
    # Build a dictionary
    my_vars = {}
    nvars = all_vars.getSize()
    # Grab the pdf we want to plot
    for i in range(0,nvars):
        p = iter.Next()
        print "%s %5.5f" % (p.GetName(), p.getVal())
        my_vars[p.GetName()] = p
    
    ################################################################################
    all_pdfs = w.allPdfs()
    iter = all_pdfs.createIterator()
    # Build a dictionary
    pdfs = {}
    npdfs = all_pdfs.getSize()
    # Grab the pdf we want to plot
    for i in range(0,npdfs):
        p = iter.Next()
        pdfs[p.GetName()] = p

    print "Grabbed all the PDF's......"



    fr_name = "fitresult_%d" % (n)
    fit_result = w.genobj(fr_name)
    #fit_result = rootfile.Get(fr_name)
    print "Printing the data and pdfs"
    print data
    print pdfs[total_name]
    fit_result.Print("v")
    nfloatpars = fit_result.floatParsFinal().getSize()
    for p in range(0,nfloatpars):
        name = fit_result.floatParsFinal()[p].GetName()
        val = fit_result.floatParsFinal()[p].getVal()
        # Set the values of the variables.
        my_vars[name].setVal(val)
        if name=="branching_fraction":
            branching_fraction_val = val

    my_vars["branching_fraction"].setVal(branching_fraction_val)
    branching_fractions.append(branching_fraction_val)

    pdfs[total_name].plotOn(frames)



    print "IN SUMMARIZE:"
    print pdfs[total_name]
    print "PDF NORM: %f" % (pdfs[total_name].getNorm())
    print "PDF VAL: %f" % (pdfs[total_name].getVal())
    nllframe,graphsll,upper_lim_vals,std_from_0 = likelihood_curve(fit_result, data,  pdfs[total_name], my_vars["branching_fraction"])

    upper_limits.append(upper_lim_vals)
    std_devs.append(std_from_0)

    print "Most likely branching_fraction: %3.3f" % (std_from_0[1])
    print "Sigma(inconsistent with 0): %3.3f" % (sqrt(2.0*std_from_0[0]))
    print "area: %3.3f" % (upper_lim_vals[0])
    print "area greater than 0: %3.3f" % (upper_lim_vals[1])
    print "ul (90%s): %3.3f" % ('%',upper_lim_vals[2])


# Write to a text file
name = "summaries_of_MC_studies/%s.txt" % (options.tag)
logfile = open(name,"w")
output = "#%11s%12s%12s\n" % ("branching_fraction","UL(90%)","DeltaLL(0)")
logfile.write(output)

for i,u in enumerate(upper_limits):
    print "-------------"
    print u
    print std_devs[i]
    print branching_fractions[i]
    output = "%12.4f%12.4f%12.4f\n" % (branching_fractions[i],u[2],std_devs[i][0])
    print output
    logfile.write(output)

logfile.close()



################################################################################
################################################################################


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


