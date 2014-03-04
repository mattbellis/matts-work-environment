#!/usr/bin/env python
################################################################################
#
# formattingExercise.py
# M. Bellis
# 06/09/09
#
# This exercise is designed to demonstrate the various formatting options in 
# ROOT and how to make the best use of them to display data cleanly and 
# efficiently. 
#
################################################################################


################################################################################
# Import some modules that we will need.
################################################################################
import sys # Helps with parsing command line options.
from math import *


################################################################################
# Parse any command line options
################################################################################
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-b", "--batch", dest="batch", action="store_true", 
        default=False, help="Run in batch mode")
parser.add_option("--num-galaxies", dest="num_galaxies", default=1000, 
        help="Number of galaxies to simulate")
parser.add_option("--num-clusters", dest="num_clusters", default=0, 
        help="Number of clusters to simulate")
parser.add_option("--cluster-width", dest="cluster_width", default=0.1, 
        help="Width of clusters (assuming Gaussian distribution.")

(options, args) = parser.parse_args()

size_of_universe = 10.0

################################################################################
# Set some global formatting options
# Upon initializtion, ROOT creates a global instance of a TStyle object.
# This allows the user to set some global formatting options that are commonly
# used, rather than setting them individually for each TCanvas, TH1F, etc.
#
# http://root.cern.ch/root/html/TStyle.html
#
################################################################################
from ROOT import * # All of our ROOT libraries.
from color_palette import *
gROOT.Reset()
gStyle.SetOptStat(11)  # What is displayed in the stats box for each histo.
gStyle.SetStatH(0.3);   # Max height of stats box
gStyle.SetStatW(0.25);  # Max height of stats box
gStyle.SetPadLeftMargin(0.20)   # Left margin of the pads on the canvas.
gStyle.SetPadBottomMargin(0.20) # Bottom margin of the pads on the canvas.
gStyle.SetFrameFillStyle(0) # Keep the fill color of our pads white.

# Better colour palette
set_palette("palette",50)

################################################################################
# Initialize a random number generator
# We use TRandom3 rather than TRandom as it has the least repeatabliity.
# We'll use this later to randomly fill some histograms.
#
# http://root.cern.ch/root/html/TRandom3.html
#
################################################################################
rnd = TRandom3()

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
num_canvases = 3
can = []
for i in range(0, num_canvases):
  name = "can%d" % (i) # Each canvas should have unique name
  title = "Data %d" % (i) # The title can be anything
  if i<2:
    can.append(TCanvas( name, title, 10+10*i, 10+10*i, 800, 800 ))
    can[i].SetFillColor( 0 )
    can[i].Divide( 1, 1 ) # Create 4 drawing pads in a 2x2 array.
  else:
    can.append(TCanvas( name, title, 10+10*i, 10+10*i, 800, 800 ))
    can[i].SetFillColor( 0 )
    can[i].Divide( 2, 2 ) # Create 2 drawing pads in a 2x1 array.


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

# Create an array of histograms
histos = []
for i in range(0,4):
    hname = "histos_%d" % (i) # Each histogram must have a unique name
    if i==0:
        histos.append(TH1F(hname,hname,1000,0,2.0*size_of_universe))
    elif i==1:
        histos.append(TH1F(hname,hname,1000,0.1,2.0*size_of_universe))
    elif i==2:
        histos.append(TH1F(hname,hname,1000,0,1.0))
    elif i==3:
        histos.append(TH1F(hname,hname,1000,0,1.0))
    else:
        histos.append(TH1F(hname,hname,200,0,4.0*size_of_universe*size_of_universe))

    # Set some formatting options
    histos[i].SetMinimum(0)
    histos[i].SetTitle("Gaussian distribution")

    histos[i].GetXaxis().SetNdivisions(6)
    histos[i].GetXaxis().SetLabelSize(0.06)
    histos[i].GetXaxis().CenterTitle()
    histos[i].GetXaxis().SetTitleSize(0.09)
    histos[i].GetXaxis().SetTitleOffset(0.8)
    histos[i].GetXaxis().SetTitle("X")

    histos[i].GetYaxis().SetNdivisions(4)
    histos[i].GetYaxis().SetLabelSize(0.06)
    histos[i].GetYaxis().CenterTitle()
    histos[i].GetYaxis().SetTitleSize(0.09)
    histos[i].GetYaxis().SetTitleOffset(1.1)
    histos[i].GetYaxis().SetTitle("Y")

    histos[i].SetFillColor(22)

################################################################################
# Create an array of 3D histograms
histos3D = []
for i in range(0,1):
    hname = "histos3D_%d" % (i) # Each histogram must have a unique name
    histos3D.append(TH3F(hname,hname,200,-size_of_universe,size_of_universe,200,-size_of_universe,size_of_universe,200,-size_of_universe,size_of_universe))

    # Set some formatting options
    histos3D[i].SetMinimum(0)
    histos3D[i].SetTitle("Gaussian distribution")

    histos3D[i].GetXaxis().SetNdivisions(6)
    histos3D[i].GetXaxis().SetLabelSize(0.06)
    histos3D[i].GetXaxis().CenterTitle()
    histos3D[i].GetXaxis().SetTitleSize(0.09)
    histos3D[i].GetXaxis().SetTitleOffset(0.8)
    xaxis_name = "%d plot" % (i+1)
    histos3D[i].GetXaxis().SetTitle( xaxis_name )

    histos3D[i].GetYaxis().SetNdivisions(4)
    histos3D[i].GetYaxis().SetLabelSize(0.06)
    histos3D[i].GetYaxis().CenterTitle()
    histos3D[i].GetYaxis().SetTitleSize(0.09)
    histos3D[i].GetYaxis().SetTitleOffset(1.1)
    histos3D[i].GetYaxis().SetTitle("# events/bin")

    histos3D[i].SetFillColor(22)

###############################################################################
###############################################################################
################################################################################
# Create an array of 2D histograms
histos2D = []
for i in range(0,1):
    hname = "histos2D_%d" % (i) # Each histogram must have a unique name
    histos2D.append(TH2F(hname,hname,200,-size_of_universe,size_of_universe,200,-size_of_universe,size_of_universe))

    # Set some formatting options
    histos2D[i].SetMinimum(0)
    histos2D[i].SetTitle("Gaussian distribution")

    histos2D[i].GetXaxis().SetNdivisions(6)
    histos2D[i].GetXaxis().SetLabelSize(0.06)
    histos2D[i].GetXaxis().CenterTitle()
    histos2D[i].GetXaxis().SetTitleSize(0.09)
    histos2D[i].GetXaxis().SetTitleOffset(0.8)
    xaxis_name = "%d plot" % (i+1)
    histos2D[i].GetXaxis().SetTitle( xaxis_name )

    histos2D[i].GetYaxis().SetNdivisions(4)
    histos2D[i].GetYaxis().SetLabelSize(0.06)
    histos2D[i].GetYaxis().CenterTitle()
    histos2D[i].GetYaxis().SetTitleSize(0.09)
    histos2D[i].GetYaxis().SetTitleOffset(1.1)
    histos2D[i].GetYaxis().SetTitle("# events/bin")

    histos2D[i].SetFillColor(22)

###############################################################################
###############################################################################
# Generate a distribution of galaxies
###############################################################################
x = []
y = []
z = []
num_galaxies = int(options.num_galaxies)
num_clusters = int(options.num_clusters)
cluster_width = float(options.cluster_width)

num_galaxies_in_cluster = 0
if num_clusters != 0:
    num_galaxies_in_cluster = int(num_galaxies/float(num_clusters))

outfile = open("default.txt","w")
while i<num_galaxies:

    # Counter
    if i%1000==0:
        print i

    if num_clusters==0:
        xpt = 2.0*size_of_universe*rnd.Rndm() - size_of_universe
        ypt = 2.0*size_of_universe*rnd.Rndm() - size_of_universe
        zpt = 2.0*size_of_universe*rnd.Rndm() - size_of_universe

        if sqrt(xpt*xpt+ypt*ypt+zpt*zpt)<size_of_universe:

            output = "%f %f %f\n" % (xpt,ypt,zpt)
            outfile.write(output)
            histos2D[0].Fill(xpt,ypt)
            histos3D[0].Fill(xpt,ypt,zpt)

            x.append(xpt)
            y.append(ypt)
            z.append(zpt)

            i += 1
    else:

        # Calculate center of cluster

        xcenter = 2.0*size_of_universe*rnd.Rndm() - size_of_universe
        ycenter = 2.0*size_of_universe*rnd.Rndm() - size_of_universe
        zcenter = 2.0*size_of_universe*rnd.Rndm() - size_of_universe

        if sqrt(xcenter*xcenter+ycenter*ycenter+zcenter*zcenter)<size_of_universe:

            for j in range(0,num_galaxies_in_cluster):

                xpt = rnd.Gaus(xcenter,cluster_width)
                ypt = rnd.Gaus(ycenter,cluster_width)
                zpt = rnd.Gaus(zcenter,cluster_width)

                output = "%f %f %f\n" % (xpt,ypt,zpt)
                outfile.write(output)

                histos2D[0].Fill(xpt,ypt)
                histos3D[0].Fill(xpt,ypt,zpt)

                x.append(xpt)
                y.append(ypt)
                z.append(zpt)

                i += 1



outfile.close()

npts = len(x)
for i in range(0,npts):

    # Counter
    if i%1000==0:
        print i

    for j in range(i+1,npts):
        deltax = x[i]-x[j]
        deltay = y[i]-y[j]
        deltaz = z[i]-z[j]
        distance = sqrt(deltax*deltax + deltay*deltay + deltaz*deltaz)
        #distance = deltax*deltax + deltay*deltay + deltaz*deltaz

        histos[0].Fill(distance)

        #wt = 1.0/(distance*distance)
        wt = 1.0/distance
        #print wt
        histos[1].Fill(distance,wt)

        histos[2].Fill(1.0/distance)
        histos[3].Fill(1.0/distance,wt)


################################################################################
# Draw the histograms 
################################################################################
can[0].cd(1) 
histos2D[0].Draw("colz")
gPad.Update()

can[1].cd(1) 
histos3D[0].Draw()
gPad.Update()

can[2].cd(1) 
histos[0].SetMinimum(1)
histos[0].Draw()
gPad.SetLogy()
gPad.Update()

can[2].cd(2) 
histos[1].SetMinimum(1)
histos[1].Draw()
gPad.SetLogy()
gPad.Update()

can[2].cd(3) 
histos[2].SetMinimum(1)
histos[2].Draw()
gPad.SetLogx()
gPad.SetLogy()
gPad.Update()

can[2].cd(4) 
histos[3].SetMinimum(1)
histos[3].Draw()
gPad.SetLogx()
gPad.SetLogy()
gPad.Update()

###############################################################################
# Save the histograms as .ps files. 
# Note that this assumes the subdirectory "Plots" exists.
###############################################################################
for i in range(0, num_canvases):
  name = "Plots/can_2pt_corr_galaxies_%d.eps" % (i)
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
