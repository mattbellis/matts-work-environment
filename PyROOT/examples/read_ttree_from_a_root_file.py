#!/usr/bin/env python
################################################################################
#
# read_ttree_from_a_root_file.py
# M. Bellis
# 10/26/09
#
# This exercise is designed to demonstrate how to read from a TTree object
# given a set of root files. This example is set up to work with BaBar data
# producted with BtaTupleMaker.
#
################################################################################


################################################################################
# Import some modules that we will need.
################################################################################
import sys
from optparse import OptionParser # Use this to parse command line options.


################################################################################
# Parse any command line options
################################################################################
parser = OptionParser()
parser.add_option("-n", "--ntuplename", dest="ntuplename", default='ntp1', 
    help="Name of the ntuple to grab (ntp1)", metavar="NTUPLENAME")
parser.add_option("-m", "--max",  dest="max", default=1e9, 
    help="Maximum number of events over which to run.")
parser.add_option("-p", "--plot_extension", dest="plot_ext", 
    help="Extension to add onto output plots")
parser.add_option("-b", "--batch", action="store_true", dest="batch", 
    default=False, help="Run in batch mode and exit.")

# Parse the options
(options, args) = parser.parse_args()

if len(sys.argv) <= 1:
  parser.error("Incorrect number of arguments")

################################################################################
# Import ROOT here. Works better with the OptionParser()
from ROOT import * # All of our ROOT libraries.
################################################################################

################################################################################
# Read in all of the files which are not arguments of other command line
# options.
################################################################################
# Create a chain based on the ntuple names
t = TChain(options.ntuplename)
# Loop over the filenames and add to tree.
for filename in args:
  print("Adding file: " + filename)
  t.Add(filename)



################################################################################
# Set some global formatting options
# Upon initializtion, ROOT creates a global instance of a TStyle object.
# This allows the user to set some global formatting options that are commonly
# used, rather than setting them individually for each TCanvas, TH1F, etc.
#
# http://root.cern.ch/root/html/TStyle.html
#
################################################################################
gROOT.Reset()
gStyle.SetOptStat(11)  # What is displayed in the stats box for each histo.
gStyle.SetStatH(0.3);   # Max height of stats box
gStyle.SetStatW(0.25);  # Max height of stats box
gStyle.SetPadLeftMargin(0.20)   # Left margin of the pads on the canvas.
gStyle.SetPadBottomMargin(0.20) # Bottom margin of the pads on the canvas.
gStyle.SetFrameFillStyle(0) # Keep the fill color of our pads white.

################################################################################
# Create some canvases on which to place our histograms.
################################################################################
#
# TCanvas constructor
# can = TCanvas( name, title, x-location of upper left corner (pixel), 
#                y-location of upper left corner (pixel), 
#                width in pixels, height in pixels )
#                
# http://root.cern.ch/root/html/TCanvas.html
#
################################################################################
num_canvases = 1
can = []
for i in range(0, num_canvases):
  name = "can%d" % (i) # Each canvas should have unique name
  title = "Data %d" % (i) # The title can be anything
  can.append(TCanvas( name, title, 10+10*i, 10+10*i, 800, 800 ))
  can[i].SetFillColor( 0 )
  can[i].Divide( 2, 2 ) # Create 4 drawing pads in a 2x2 array.


################################################################################
# Create some empty histograms and set some basic formatting options.
################################################################################
#
# TH1F constructor
# h = TH1F ( name, title, number of bins, lower edge of minimum bin, 
#            highest edge of maximum bin )
#
# Note that if you wanted 10 equal bins that would contain the numbers 1 to 10, 
# your constructor would be
# 
# h = TH1F ( name, title, 10, 0, 10)
# 
# *not*
#
# h = TH1F ( name, title, 10, 1, 10)
#
# http://root.cern.ch/root/html/TH1.html
#
# http://root.cern.ch/root/html/TH1F.html
#
################################################################################

# Create a 2x4 array of histograms
histos = []
for i in range(0,2):
  histos.append([])
  for j in range(0,4):
    hname = "histos%d_%d" % (i, j) # Each histogram must have a unique name
    htitle = "Histogram %d %d" % (i, j) # Give each its own title.
    if (j == 0):
      htitle = "m_{ES}"
      histos[i].append( TH1F(hname, htitle, 10, 0.5, 10.5) )
    elif (j == 1):
      htitle = "#Delta E"
      histos[i].append( TH1F(hname, htitle, 10, -5.0, 5.0) )
    else:
      htitle = "Physics"
      histos[i].append( TH1F(hname, htitle, 10, -5.0, 5.0) )

    # Set some formatting options
    histos[i][j].SetMinimum(0)
    
    histos[i][j].GetXaxis().SetNdivisions(6)
    histos[i][j].GetXaxis().SetLabelSize(0.06)
    histos[i][j].GetXaxis().CenterTitle()
    histos[i][j].GetXaxis().SetTitleSize(0.09)
    histos[i][j].GetXaxis().SetTitleOffset(0.8)
    histos[i][j].GetXaxis().SetTitle( "Some distribution" )

    histos[i][j].GetYaxis().SetNdivisions(4)
    histos[i][j].GetYaxis().SetLabelSize(0.06)
    histos[i][j].GetYaxis().CenterTitle()
    histos[i][j].GetYaxis().SetTitleSize(0.09)
    histos[i][j].GetYaxis().SetTitleOffset(1.1)
    histos[i][j].GetYaxis().SetTitle("# events/bin")

################################################################################
# Loop over our TChain entries and fill the histograms
################################################################################
t.SetBranchStatus("*",0) # Turn off all the entries
#t.SetBranchStatus("*",1) # Turn on all the entries
t.SetBranchStatus("nB",1) # Turn on an individual entry.
t.SetBranchStatus("BpostFitMes",1) # Turn on an individual entry.
t.SetBranchStatus("BpostFitDeltaE",1) # Turn on an individual entry.

# Get number of enries and make sure it's not greater than the max number
# we passed in on the command line.
nentries = t.GetEntries()
if nentries==0: # Break if empty TTree/TChain
  print "Empty files!"
  sys.exit(1)

print "Entries: %d" % (nentries)
if int(options.max) < nentries:
  nentries = int(options.max)

# Loop over the entries
for n in xrange (nentries):
  if n % 10000 == 0:
    print "Event number %d out of %d " % (n, nentries)

  # Grab the n'th entry
  t.GetEntry(n)

  nB = t.nB # Get the number of B candidates.

  # Fill the histograms for each B candidate
  for i in xrange(nB):
    histos[0][0].Fill( t.BpostFitMes[i] )
    histos[0][1].Fill( t.BpostFitDeltaE[i] )


################################################################################
# Draw the histograms 
################################################################################
for i in range(0,1):
  for j in range(0,4):
    can[i].cd(j + 1) # Change to the (i+1)th pad. Note the numbering starts at 1.
    histos[i][j].Draw()

    # This must be called to get the plots to display properly.
    gPad.Update()


###############################################################################
# Save the histograms as some sort of files, if we have specified a file type
# in the command line options (ps, eps, jpg, etc). 
# Note that this assumes the subdirectory "Plots" exists.
###############################################################################
if options.plot_ext:
  for i in range(0, num_canvases):
    name = "Plots/can_formatting_exercise_reading_histos_%d.%s" % (i, options.plot_ext)
    can[i].SaveAs(name)

# Delete the TChain object
del t

###############################################################################
# Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
###############################################################################
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
