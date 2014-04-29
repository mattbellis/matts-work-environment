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
parser.add_option("--ntp", dest="ntp", default="ntp1", help="Baryon [LambdaC, Lambda0]")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--no-gc", dest="no_gc", action = "store_true", default=False, help="Don't use the gaussian constraint")
#parser.add_option("--branching-fraction", dest="branching_fraction", default=0.0, help="Branching fraction.")
parser.add_option("--num-sig", dest="num_sig", help="Number of embeded signal events")
parser.add_option("--num-bkg", dest="num_bkg", default=650, help="Number of background events")
parser.add_option("--num-bins", dest="num_bins", default=50, help="Number of bins to use")
#parser.add_option("-n", "--num-fits", dest="num_fits", default=10, help="Number of toy studies in the file")
parser.add_option("--use-double-cb", dest="use_double_cb", action = "store_true", default = False, help="Use the double CB in Delta E")
parser.add_option("--starting-vals-file", dest="starting_vals_file", default="default_starting_vals_file.txt", help="File to use for the starting values")
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

outfilename = "Extracted_toy_MC_results/%s.txt" % (options.tag)
outfile = open(outfilename,"w")

starting_values = [[], []]
if options.starting_vals_file != None:
  infile = open(options.starting_vals_file)
  for line in infile:
    if line[0] != "#":
      start_val = line.split()
      starting_values[0].append(start_val[0])
      starting_values[1].append(start_val[2])

################################################
from file_map import *

pass_info = fit_pass_info(options.baryon, options.ntp, int(options.my_pass))

conv_factor_calc = float(pass_info[8][0])
conv_factor_err =  float(pass_info[8][1])

################################################
################################################

#num_pars = 2
#nfits = int(options.num_fits)

########################################
# Try running a fit
########################################

rootfile = TFile(options.results_filename, "READ")
#rootfile.ls()
keys = rootfile.GetListOfKeys()
nfits = rootfile.GetNkeys()

# Count how many keys are actually fitresult objects
nfits = 0
for k in keys:
    #print k.GetName()
    if k.GetName().find("fitresult")>=0:
        nfits += 1


# Get the starting/true vals
true_vals = []
name = "fitresult_0" 
fitresult = gROOT.FindObject(name)
fitresult.Print("v")
#exit(-1)
num_pars = fitresult.floatParsFinal().getSize()

print "num_pars: %d" % (num_pars)


par_names = []
tot_vals = []
tot_errs = []
true_nsig = 0
true_nsig = float(options.num_sig)
true_bf = 0
for j in range(0, num_pars):
  tot_vals.append(0.0)
  tot_errs.append(0.0)
  par = fitresult.floatParsFinal()[j]
  name = par.GetName()
  par_names.append(name)
  tv = -999.0
  if name == "nsig":
    tv = float(options.num_sig)
  elif name == "nbkg":
    tv = float(options.num_bkg)
  elif name == "conv_factor_fit":
    tv = conv_factor_calc
  elif name == "branching_fraction":
    tv = float(options.num_sig)/conv_factor_calc
    true_bf = tv
  else:
    if name in starting_values[0]:
      ind = starting_values[0].index(name)
      tv = float(starting_values[1][ind])

  true_vals.append(tv)

#print starting_values
#print true_vals

# Make the histos
colors = [1, 2, 4]
h = []
for j in range(0, num_pars):
  h.append([])
  for k in range(0, 3): # vals, errs, pulls
    name = "h%d_%d" % (j,k)
    width = 4.0*sqrt(abs(true_vals[j]))
    plot_tag = "default"
    if k==0:
      plot_tag = "fit vals"
      xlo = true_vals[j]-width
      xhi = true_vals[j]+width
    elif k==1:
      mid = sqrt(abs(true_vals[j]))
      xlo = mid - 5.0
      xhi = mid + 5.0
      plot_tag = "fit errs"
    elif k==2:
      xlo = -5.0
      xhi =  5.0
      plot_tag = "pulls"
    if true_vals[j]<0.0:
      dum = xlo
      xlo = xhi
      xhi = dum
    h[j].append(TH1F(name, name, int(options.num_bins), xlo, xhi))
    #print "%s %f %f" % (par_names[j], xlo, xhi)

    h[j][k].SetMinimum(0)
    h[j][k].SetTitle("")

    h[j][k].GetXaxis().SetNdivisions(6)
    h[j][k].GetXaxis().SetLabelSize(0.06)
    h[j][k].GetXaxis().CenterTitle()
    h[j][k].GetXaxis().SetTitleSize(0.09)
    h[j][k].GetXaxis().SetTitleOffset(0.8)
    xaxis_title = "%s %s" % (par_names[j], plot_tag)
    h[j][k].GetXaxis().SetTitle( xaxis_title )

    h[j][k].GetYaxis().SetNdivisions(4)
    h[j][k].GetYaxis().SetLabelSize(0.06)
    h[j][k].GetYaxis().CenterTitle()
    h[j][k].GetYaxis().SetTitleSize(0.09)
    h[j][k].GetYaxis().SetTitleOffset(1.1)
    h[j][k].GetYaxis().SetTitle("# events/bin")

    h[j][k].SetLineColor(colors[k])
    h[j][k].SetLineWidth(2)



# Make the canvases
can = []
ncans = num_pars + 1
for i in range(0,ncans):
  name = "can%d" % (i)
  can.append(TCanvas(name, name, 10+10*i, 10+10*i, 1200, 350))
  can[i].SetFillColor(0)
  if i<num_pars:
    can[i].Divide(3,1)
  else:
    can[i].Divide(1,1)

#mcstudy = gROOT.FindObject("testmcstudy")
pulls = []
r = []
num_bad_fits = 0
print "nfits: %d" % (nfits)
for i in range(0, nfits):
  # Get the fit result
  name = "fitresult_%d" % (i)
  #print name
  r.append(gROOT.FindObject(name))
  #r[i].Print("v")
  for j in range(0, num_pars):
    par = r[i].floatParsFinal()[j]

    name = par.GetName()
    val = par.getVal() 
    true_val = true_vals[j]

    delta = val - true_val
    pull = -999
    err = -999

    if par.hasAsymError():
      if delta<0:
        err = par.getAsymErrorHi() 
        pull = (delta)/err
      else:
        err = par.getAsymErrorLo() 
        pull = (-delta)/err
    else:
      err = par.getError()
      pull = (delta)/err

    tot_vals[j] += val
    tot_errs[j] += err

    """
    if j==0:
      print "%f %s %f %f %f %f" % ( r[i].minNll() , name, val , err, true_val, pull)
    """

    #print "status: %d\t covQual: %d" % (r[i].status(), r[i].covQual())
    #if r[i].minNll()>-1e6:
    if r[i].covQual()==3:
      h[j][0].Fill(val)
      h[j][1].Fill(err)
      h[j][2].Fill(pull)
    else:
      if j==0:
        #print "%f" % ( r[i].minNll() )
        num_bad_fits += 1

#print "Fit val mean: %f" % ( tot_val/float(options.num_fits) )
#print "Fit err mean: %f" % ( tot_err/float(options.num_fits) )


###############################################
# Draw everything
###############################################
f1 = TF1("f1", "gaus", -5.0, 5.0);
mean_text = []
means = []
mean_errs = []
sigmas = []
output = "%-20s  %10s %8s %8s %15s %15s %12s %12s %12s %12s\n" % \
        ("# Name","true_bf", "truesig", "true_val","mean_vals","mean_errs","mean_pull",\
        "mean_err_pull","sigma_pull","sigma_err_pull")
outfile.write(output)
for j in range(0, num_pars):
  mean_text.append([])
  output = "%-20s %10.3f %8.1f %8.4f" % (par_names[j], true_bf, true_nsig, true_vals[j])
  for k in range(0,3):
    npad = 1 + k
    can[j].cd(npad)
    h[j][k].Draw("e")

    mean_text[j].append(TPaveText(0.60, 0.70, 0.99, 0.99, "NDC"))

    text = "default"
    if k<2:
      mu = -1.0
      if k==0:
        mu = tot_vals[j]/float(nfits)
      elif k==1:
        mu = tot_errs[j]/float(nfits)
      text = "#mu: %2.2f" % (mu)
      mean_text[j][k].AddText(text)
      output += "%21.4f " % (mu)

    if k==2:
      h[j][k].Fit("f1","EQR","",-5, 5)
      mean      = f1.GetParameter(1)
      sigma     = f1.GetParameter(2)
      mean_err  = f1.GetParError(1)
      sigma_err = f1.GetParError(2)
      #print mean
      #print mean_err
      means.append(mean)
      mean_errs.append(mean_err)
      sigmas.append(sigma)

      text = "#mu: %2.3f #pm %2.3f" % (mean, mean_err)
      mean_text[j][k].AddText(text)
      output += "%12.3f %12.3f" % (mean, mean_err)

      text = "#sigma: %2.3f #pm %2.3f" % (sigma, sigma_err)
      mean_text[j][k].AddText(text)
      output += "%12.3F %12.3f" % (sigma, sigma_err)

      h[j][k].Draw("e*")

    mean_text[j][k].SetFillColor(1)
    mean_text[j][k].SetTextColor(0)
    mean_text[j][k].SetFillStyle(1001)
    mean_text[j][k].Draw()

    gPad.Update()

  output += "\n"
  outfile.write(output)
  # Save the nbad fits


output = "%-20s %10.3f %8.1f %8.1f %21.1f %15s %12s %12s %12s %12s\n" % \
        ("nbadfits",true_bf, true_nsig, nfits, num_bad_fits,"0","0",\
        "0","0","0")
outfile.write(output)

print "# bad fits: %d" % ( num_bad_fits )

################################################################################
# Make the summary plots
################################################################################
hsummary = TH1F("hsummary", "", num_pars, 0.5, num_pars+0.5)
for i in range(0,num_pars):
  hsummary.Fill(par_names[i], means[i])
  print mean_errs[i]
  hsummary.SetBinError(i+1, mean_errs[i])

can[num_pars].cd(1)
hsummary.GetYaxis().SetRangeUser(-0.2, 0.2)
hsummary.Draw("e")
line = TLine(0.0, 0.0, num_pars+1.0, 0.0)
line.SetLineStyle(2)
line.SetLineColor(6)
line.Draw()
gPad.Update()
################################################################################
################################################################################

for j in range(0, num_pars):
  can[j].cd()
  can[j].Update()
  name = "Plots/extracted_mcstudies_%s_%d.eps" % (options.tag, j)
  can[j].SaveAs(name)


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


