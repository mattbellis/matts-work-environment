#!/usr/bin/env python
################################################################################
#
# read_and_plot_from_a_text_file.py
# M. Bellis
# 06/11/09
#
# This exercise is designed to demonstrate how to read in information from 
# a text file and make histograms of that data.
#
################################################################################


################################################################################
# Import some modules that we will need.
################################################################################
import sys # Helps with parsing command line options.
from ROOT import TH1F
from ROOT import TLorentzVector
from ROOT import TCanvas
from ROOT import gROOT
from ROOT import gStyle
from ROOT import gPad

################################################################################
# Parse any command line options
################################################################################
# Make sure there is a text file passed in as the first argument
infile_name = "default.txt"
if len(sys.argv) < 2:
    print "\n\nUsage: read_and_plot_from_a_text_file.py <filename>\n\n"
    sys.exit(-1)
else:
    infile_name = sys.argv[1] 

# Open the input file
infile = open(infile_name)

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
gStyle.SetOptStat(1110)  # What is displayed in the stats box for each histo.
gStyle.SetStatH(0.2);   # Max height of stats box
gStyle.SetStatW(0.35);  # Max height of stats box
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
num_canvases = 2
can = []
for i in range(0, num_canvases):
    name = "can%d" % (i) # Each canvas should have unique name
    title = "Data %d" % (i) # The title can be anything
    if i<1:
        can.append(TCanvas( name, title, 10+10*i, 10+10*i, 800, 600 ))
        can[i].SetFillColor( 0 )
        can[i].Divide( 2, 2 ) # Create 4 drawing pads in a 2x2 array.
    else:
        can.append(TCanvas( name, title, 10+10*i, 10+10*i, 800, 400 ))
        can[i].SetFillColor( 0 )
        can[i].Divide( 2, 1 ) # Create 2 drawing pads in a 2x1 array.


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

# A few utility variables
line_colors = [ 1, 2, 4, 8 ]

# Create a 2x4 array of histograms
num_variables = 6

histos = []
for i in range(0, num_variables):
    hname = "histos%d" % (i,) # Each histogram must have a unique name
    if i == 0:
        histos.append(TH1F(hname,hname,100, 0.0, 10.0))
        histos[i].SetTitle("Initial state mass")
    elif i == 1:
        histos.append(TH1F(hname,hname,100,0.0,10.0))
        histos[i].SetTitle("0 mass")
    elif i == 2:
        histos.append(TH1F(hname,hname,200,0.0,10.0))
        histos[i].SetTitle("1 mass")
    elif i == 3:
        histos.append(TH1F(hname,hname,200,0.0,10.0))
        histos[i].SetTitle("2 mass")
    elif i == 4:
        histos.append(TH1F(hname,hname,150,0.0,10.0))
        histos[i].SetTitle("0+1 mass")
    elif i == 5:
        histos.append(TH1F(hname,hname,150,0.0,10.0))
        histos[i].SetTitle("intial-0-1 mass")

    # Set some formatting options
    histos[i].SetMinimum(0)

    histos[i].GetXaxis().SetNdivisions(6)
    histos[i].GetXaxis().SetLabelSize(0.06)
    histos[i].GetXaxis().CenterTitle()
    histos[i].GetXaxis().SetTitleSize(0.09)
    histos[i].GetXaxis().SetTitleOffset(0.8)
    histos[i].GetXaxis().SetTitle( "M (GeV/c^{2})" )

    histos[i].GetYaxis().SetNdivisions(4)
    histos[i].GetYaxis().SetLabelSize(0.06)
    histos[i].GetYaxis().CenterTitle()
    histos[i].GetYaxis().SetTitleSize(0.09)
    histos[i].GetYaxis().SetTitleOffset(1.1)
    histos[i].GetYaxis().SetTitle("# events/bin")

    histos[i].SetFillColor(7)

###############################################################################
# Read each line of the file and grab some variables from that line. 
# Fill some histograms with this information.
###############################################################################
initial_particle = TLorentzVector()
final_particle_0 = TLorentzVector()
final_particle_1 = TLorentzVector()
final_particle_2 = TLorentzVector()

# Read the first line of the file
line = infile.readline()

# Loop over all the lines in the file
while not line == "":

    # The first line in the file, and the event delimiter is the number of 
    # particles in a given event.
    nparticles = int(line.split()[0])

    # Loop over all the particles and fill some histograms.
    for i in range(0,nparticles):

        line = infile.readline()
        vals = line.split()

        E = float(vals[0])
        x = float(vals[1])
        y = float(vals[2])
        z = float(vals[3])

        if i==0:
            initial_particle.SetXYZT(x, y, z, E)
        elif i==1:
            final_particle_0.SetXYZT(x, y, z, E)
        elif i==2:
            final_particle_1.SetXYZT(x, y, z, E)
        elif i==3:
            final_particle_2.SetXYZT(x, y, z, E)

        histos[0].Fill(initial_particle.M())
        histos[1].Fill(final_particle_0.M())
        histos[2].Fill(final_particle_1.M())
        histos[3].Fill(final_particle_2.M())

        histos[4].Fill((final_particle_0 + final_particle_1).M())
        histos[5].Fill((initial_particle - final_particle_0 - final_particle_1).M() )

    # Make sure you read in the next line before you go back to the top of the 
    # loop.
    line = infile.readline()

################################################################################
# Draw the histograms 
################################################################################
# Plot the first 4 histograms.
for i in range(0,4):
  can[0].cd(i+1) # Change to the (i+1)th pad. Note the numbering starts at 1.
  histos[i].Draw()
  gPad.Update()

# Plot the next two histograms.
for i in range(0,2):
  can[1].cd(i+1) # Change to the (i+1)th pad. Note the numbering starts at 1.
  histos[i+4].Draw()
  gPad.Update()

###############################################################################
# Save the histograms as .ps files. 
# Note that this assumes the subdirectory "Plots" exists.
###############################################################################
for i in range(0, num_canvases):
  name = "Plots/can_formatting_exercise_%d.eps" % (i)
  can[i].SaveAs(name)

###############################################################################
# Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
###############################################################################
if __name__ == '__main__':
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]
