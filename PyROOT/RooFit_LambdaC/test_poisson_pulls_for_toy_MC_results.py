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
parser.add_option("--results-file", dest="results_filename", default="default_results.root", help="File from which to read the results.")
parser.add_option("--num-embed", dest="num_embed", help="Number of embeded signal events")
parser.add_option("--num-bkg", dest="num_bkg", default=650, help="Number of background events")
parser.add_option("--num-bins", dest="num_bins", default=25, help="Number of bins to use")
parser.add_option("-n", "--num-fits", dest="num_fits", default=10, help="Number of toy studies in the file")
parser.add_option("--use-double-cb", dest="use_double_cb", action = "store_true", default = False, help="Use the double CB in Delta E")
parser.add_option("--starting-vals-file", dest="starting_vals_file", default="default_starting_vals_file.txt", help="File to use for the starting values")
parser.add_option("--true-vals-file", dest="true_vals_file", default="num_count.txt", help="File to use for the true sig/bkg values")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")

(options, args) = parser.parse_args()

import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

from backgroundAndSignal_NEW_def import *
import backgroundAndSignal_NEW_def 


#######################################################
gROOT.Reset()
gStyle.SetOptStat(11111)
gStyle.SetOptFit(111111)
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

################################################
################################################

true_vals = []
if options.true_vals_file != None:
  infile = open(options.true_vals_file)
  for line in infile:
    #print line
    if line[0] != "#":
      true_vals.append(line.split())

################################################
################################################

frames = []

########################################
# Try running a fit
########################################

#lo = 0.9
#hi = 1.005
#rpsf_s, rpsf_b, sig_pdf, bkg_pdf, fit_func = tot_PDF( dim, options.use_double_cb, lo, hi )

#sig_pdf, bkg_pdf, fit_func = tot_PDF( dim, options.use_double_cb )

rootfile = TFile(options.results_filename, "READ")
#rootfile.ls()

h = []
for i in range(0,1):
  name = "h%d" % (i)
  h.append(TH1F(name, name, 100, -5.0, 4.5))



#mcstudy = gROOT.FindObject("testmcstudy")
tot_val = 0.0
tot_err = 0.0

true_index = 2
pulls = []
r = []
pval = 1
nfits = int(options.num_fits)
for i in range(0, nfits):
  name = "fitresult_%d" % (i)
  r.append(gROOT.FindObject(name))
  name = r[i].floatParsFinal()[pval].GetName()
  val = r[i].floatParsFinal()[pval].getVal() 
  #err = r[i].floatParsFinal()[pval].getError() 

  true_val = float(true_vals[nfits - i - 1][true_index]) - float(true_vals[nfits - i - 1][0])
  if options.num_embed != None:
    true_val = float(options.num_embed)

  delta = val - true_val
  pull = -999
  err = -999
  par = r[i].floatParsFinal()[pval]
  if par.hasAsymError():
    if delta<0:
      err = par.getAsymErrorHi() 
      #print err
      pull = (delta)/err
    else:
      err = par.getAsymErrorLo() 
      #print err
      pull = (-delta)/err
  else:
    err = par.getError()
    #err = 6.0
    pull = (delta)/err

  tot_val += val
  tot_err += err

  #print "%f %s %f %f %f %f" % ( r[i].minNll() , name, val , err, true_val, pull)
  h[0].Fill(pull)


can = TCanvas("can", "can", 10, 10, 900, 900)
can.SetFillColor(0)
can.Divide(1,1)

can.cd(1)
h[0].Draw("e*")
h[0].Fit("gaus","EQR","",-5, 4.5)
h[0].Draw("e*")
gPad.Update()

print "Fit val mean: %f" % ( tot_val/float(options.num_fits) )
print "Fit err mean: %f" % ( tot_err/float(options.num_fits) )



## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


