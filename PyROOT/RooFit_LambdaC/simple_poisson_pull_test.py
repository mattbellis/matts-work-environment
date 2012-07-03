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
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("--num-embed", dest="num_embed", default=0, help="Number of embeded signal events")
parser.add_option("--num-sig", dest="num_sig", default=50, help="Number of signal events")
parser.add_option("--num-bins", dest="num_bins", default=25, help="Number of bins to use")
parser.add_option("-n", "--num-fits", dest="num_fits", default=1000, help="Number of toy studies in the file")
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

num_bins = 25
if options.num_bins:
  num_bins = int(options.num_bins)

################################################
################################################

rnd = TRandom3()

lo = float(options.num_sig) - 0.6*float(options.num_sig)
hi = float(options.num_sig) + 0.6*float(options.num_sig)

h = []
for i in range(0,4):
  name = "h%d" % (i)
  if i<2:
    h.append(TH1F(name, name, 50, lo, hi))
  else:
    h.append(TH1F(name, name, 50, -5.0, 5.0))


total_fit = 0.0
total_org = 0.0
# gen the numbers
org_vals = []
fit_vals = []
fit_err = 0.2*float(options.num_sig)
errs = []
for n in xrange(int(options.num_fits)):
  num = rnd.Poisson( int(options.num_sig) )
  fit = num + rnd.Gaus(0.0, fit_err)
  err = rnd.Gaus(0.0, fit_err)
  org_vals.append(num)
  fit_vals.append(fit)
  errs.append(err)
  h[0].Fill(num)
  h[1].Fill(fit)
  total_org += num
  total_fit += fit

#mcstudy = gROOT.FindObject("testmcstudy")
true_index = 2
pulls = []
r = []
pval = 11
nfits = int(options.num_fits)
for n,v in enumerate(org_vals):
  pull = ( fit_vals[n] - v )/fit_err
  h[2].Fill(pull)
  #pull = (fit_vals[n] - float(options.num_sig))/fabs(errs[n])
  #pull = (fit_vals[n] - float(options.num_sig))/sqrt(fabs(fit_err))
  pull = (fit_vals[n] - float(options.num_sig))/(fit_err*1.4)
  h[3].Fill(pull)

print total_org
print total_fit
print "Mean org: %f" % (total_org/float(options.num_fits))
print "Mean fit: %f" % (total_fit/float(options.num_fits))

can = TCanvas("can", "can", 10, 10, 900, 900)
can.SetFillColor(0)
can.Divide(2,2)

can.cd(1)
h[0].Draw()
gPad.Update()

can.cd(2)
h[1].Draw()
gPad.Update()


can.cd(3)
h[2].Draw()
h[2].Fit("gaus")
gPad.Update()


can.cd(4)
h[3].Draw()
h[3].Fit("gaus")
gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


