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
import os
from optparse import OptionParser

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--num-fits", dest="num_fits", default=100, help="Number of fits to run")
parser.add_option("--num-sig", dest="num_sig", default=0, help="Number of signal events, embedded or otherwise.")
parser.add_option("--num-bkg", dest="num_bkg", default=650, help="Number of background events")
parser.add_option("--num-bins", dest="num_bins", default=25, help="Number of bins to use")
parser.add_option("--use-double-cb", dest="use_double_cb", action = "store_true", default = False, help="Use the double CB in Delta E")
parser.add_option("--pure", dest="pure", action = "store_true", default = False, help="Do pure internal fits.")
parser.add_option("--starting-vals-file", dest="starting_vals_file", default="default_starting_vals_file.txt", help="File to use for the starting values")
#parser.add_option("--true-vals-file", dest="true_vals_file", default="default_true_vals_file.txt", help="File to use for the true sig/bkg values")
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
#gStyle.SetOptStat(110010)
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

starting_values = []
if options.starting_vals_file != None:
  infile = open(options.starting_vals_file)
  for line in infile:
    if line[0] != "#":
      start_val = line.split()
      starting_values.append([start_val[0], start_val[2], int(start_val[3]) ] )

"""
if options.true_vals_file != None:
  infile = open(options.true_vals_file)
  for line in infile:
    if line[0] != "#":
      true_vals = line.split()
"""
#print starting_values

################################################
################################################


frames = []

########################################
# Try running a fit
########################################
######################################
# Set the background variables
######################################
##################################################
lo = psf_lo
hi = psf_hi-0.00001

rs_dum = myRooParSF("sig", lo, hi)
rb_dum = myRooParSF("bkg", lo, hi)
mypars = get_mypars()
# Set the starting values from the file
fitvars = []
for s in starting_values:
  for p in mypars:
    if s[0] == p.GetName():
      p.setVal(float(s[1]))
      p.setConstant(s[2])
      if s[2] == 0:
        fitvars.append(p)

# Try setting reasonable values
"""
for p in mypars:
  if p.GetName().find("binHeight") == -1:
    p.setMax( 10.0*abs(p.getVal()) )
"""

##############################################
# Set num sig/bkg by hand
##############################################
nbkg.setVal( float(options.num_bkg) )
nsig.setVal( float(options.num_sig) )

#nbkg.setMin(0.0)
#nbkg.setMax(10000.0)
#nsig.setMin(0.0)
#nsig.setMax(2000.0)

#argpar.setMin(-50.0)
#argpar.setMax(0.0)
#p1.setMin(-5.0)
#p1.setMax(0.0)
#"""


# Grab the fit functions which we will need.
rpsf_s, rpsf_b, sig_pdf, bkg_pdf, fit_func = tot_PDF( dim, options.use_double_cb, lo, hi )

#########################################################
# Run the fit
#########################################################
x.setRange("FULL",5.2, 5.30)
y.setRange("FULL",-0.2, 0.2)
z.setRange("FULL", psf_lo, psf_hi)

x.setRange("SIGNAL",5.25, 5.30)
y.setRange("SIGNAL",-0.10, 0.1)
z.setRange("SIGNAL", psf_lo, psf_hi)

# Sideband 1 region
x.setRange("SB1", 5.2,  5.30)
y.setRange("SB1", 0.075, 0.2)
z.setRange("SB1", psf_lo, psf_hi)
# Sideband 2 region
x.setRange("SB2",  5.2,  5.30)
y.setRange("SB2", -0.2, -0.075)
z.setRange("SB2", psf_lo, psf_hi)
# Sideband 3 region
x.setRange("SB3",  5.2,  5.27)
y.setRange("SB3", -0.075, 0.075)
z.setRange("SB3", psf_lo, psf_hi)



##################################
#
# Define the fit function
#
##################################
x.setBins(40)
y.setBins(40)
z.setBins(40)

rooargs = RooArgSet(x,y,z)
if dim == 2:
  rooargs = RooArgSet(x,y)

mcstudy = RooMCStudy(fit_func, rooargs, \
    RooCmdArg(RooFit.Extended(kTRUE)), \
    RooCmdArg(RooFit.Silence()), \
    RooCmdArg( RooFit.FitOptions( RooFit.Save(kTRUE), RooFit.PrintEvalErrors(-1), RooFit.Range("FULL") )))

"""
mcstudy = RooMCStudy(fit_func, rooargs, \
    RooCmdArg(RooFit.Extended(kTRUE)), \
    RooCmdArg(RooFit.Silence()), \
    #RooCmdArg(RooFit.Binned(kTRUE)), \
    #RooCmdArg(RooFit.FitOptions(RooFit.Save(kTRUE), RooFit.PrintEvalErrors(-1) )))
    #RooCmdArg(RooFit.FitOptions(RooFit.Save(kTRUE), RooFit.PrintEvalErrors(-1), RooFit.Strategy(2), Math.Minimizer("Minuit","migradimproved"))))
    RooCmdArg(RooFit.FitOptions(RooFit.Save(kTRUE), RooFit.PrintEvalErrors(-1), RooFit.Strategy(2) )))
"""



###############################################
#
# Set up and run the fits
#
###############################################
dirname = options.tag
if not os.path.exists (dirname):
  os.makedirs (dirname)

# Define the input files for embedded fits
if not options.pure:
  infilename = "%s/mcstudies_bkg%d_embed_sig%d" % ( dirname, int(options.num_bkg), int(options.num_sig) )
  if int(options.num_sig) == 0:
    infilename = "%s/mcstudies_bkg%d" % ( dirname, int(options.num_bkg) )
  infilename += "_%04d.dat"
  print infilename

  mcstudy.fit( int(options.num_fits) , infilename )

else:
  # And for pure studies
  outfilename = "%s/mcstudies_pure_bkg%d_sig%d" % ( dirname, int(options.num_bkg), int(options.num_sig) )
  outfilename += "_%04d.dat"
  print outfilename

  mcstudy.generateAndFit( int(options.num_fits), int(options.num_bkg)+int(options.num_sig), True , outfilename)




#mcstudy.generateAndFit(200)

#vars = [argpar, cutoff, p1, nbkg]
vars = [argpar, p1, nbkg, nsig]
vars = fitvars
print vars
for v in vars:
  print "%s %f" % ( v.GetName(), v.getMax() )

frames = [ [], [], [] ]
for j in range(0, len(vars) ):
  frames[0].append(mcstudy.plotParam(vars[j], RooFit.Bins(100) ) )
  frames[1].append(mcstudy.plotError(vars[j], RooFit.Bins(100) ) )
  frames[2].append(mcstudy.plotPull(vars[j], RooFit.Bins(100), RooCmdArg(RooFit.FitGauss(kTRUE)) ) )
  for i in range(0,3):
    frames[i][j].SetMinimum(0)
    #frames[i][j].SetMaximum(0.25 * float(options.num_fits))
    name = "frame%d_%d" % ( i, j )
    frames[i][j].SetName(name)

    frames[i][j].GetYaxis().SetNdivisions(4)
    frames[i][j].GetXaxis().SetNdivisions(4)

    frames[i][j].GetYaxis().SetLabelSize(0.06)
    frames[i][j].GetXaxis().SetLabelSize(0.06)

    frames[i][j].GetXaxis().CenterTitle()
    frames[i][j].GetXaxis().SetTitleSize(0.06)
    frames[i][j].GetXaxis().SetTitleOffset(1.0)

    frames[i][j].GetYaxis().CenterTitle()
    frames[i][j].GetYaxis().SetTitleSize(0.06)
    frames[i][j].GetYaxis().SetTitleOffset(1.1)

    frames[i][j].SetMarkerSize(0.01)



# Plot data and PDF overlaid
cantoy = []
for i in range(0, 3):
  name = "cantoy%d" % (i)
  cantoy.append(TCanvas(name, name, 20+10*i, 20+10*i, 1300, 900))
  cantoy[i].SetFillColor(0)
  n = int(sqrt(float(len(vars)))) + 1
  cantoy[i].Divide(n, n)

mean_text = []
for i in range(0,3):
  mean_text.append([])
  for j in range(0, len(vars) ):
    cantoy[i].cd(j + 1)
    #"""
    if i>1:
      frames[i][j].GetXaxis().SetLimits(-5.0, 5.0)
    #"""

    #"""
    if not(int(options.num_sig) == 0 and j==3):
      frames[i][j].GetYaxis().SetRangeUser(0.0, 0.15*int(options.num_fits))
    #"""

    frames[i][j].Draw()
    ypts = ROOT.Double(0.0)
    xpts = ROOT.Double(0.0)
    tot = 0.0
    count = 0
    if i<2:
      ####################################
      # Calc the mean of some of the plots
      ####################################
      npts = frames[i][j].getHist().GetN()
      for n in range(0, npts):
        frames[i][j].getHist().GetPoint(n, xpts, ypts)
        tot += xpts * ypts
        count += ypts
      print "%f %f %f %d" % ( xpts , ypts , tot , count)
      mu = -1.0
      print count
      if count==0:
        mu = -1.0
      else:
        mu = tot/count

      #mu = frames[i][j].getHist().GetMean()
      mean_text[i].append(TPaveText(0.70, 0.80, 0.99, 0.99, "NDC"))
      name = "#mu: %2.2f" % (mu)
      mean_text[i][j].AddText(name)
      mean_text[i][j].SetFillColor(1)
      mean_text[i][j].SetTextColor(0)
      mean_text[i][j].SetFillStyle(1001)
      mean_text[i][j].Draw()

    gPad.Update()


#########################################################
# Print the canvases
#########################################################
for i in range(0, 3):

  cbstring = ""
  if options.use_double_cb:
    cbstring = "doubleCB_"

  tagstring = "embedded_"
  if options.pure:
    tagstring = "pure_"

  name = "Plots/cantoy_fit_%s%strials%d_bkg%d_sig%d_%s_%d.eps" % ( tagstring, cbstring, int(options.num_fits), int(options.num_bkg), int(options.num_sig), options.tag, i)

  cantoy[i].SaveAs(name)

#########################################################
# Save the mc study
#########################################################
name = "toy_results_%s_trials%d_bkg%d_sig%d.root" % (options.tag, int(options.num_fits), int(options.num_bkg), int(options.num_sig))
rootoutfile = TFile(name, "RECREATE")
print "Writing output file: %s" % (name)
#mcstudy.Write("testmcstudy")
#"""
for i in range(0, int(options.num_fits)):
  name = "fitresult_%d" % (i)
  mcstudy.fitResult(i).Write(name)
#"""
rootoutfile.Close()




## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


