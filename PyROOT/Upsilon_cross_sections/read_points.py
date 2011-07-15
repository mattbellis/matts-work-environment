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
from array import *

import sys # Helps with parsing command line options.
from ROOT import * # All of our ROOT libraries.

################################################################################
# Parse any command line options
################################################################################
# Make sure there is a text file passed in as the first argument
infile_name = "default.txt"
if len(sys.argv) < 2:
  print "\n\nUsage: read_and_plot_from_a_text_file.py <filename> [max events to process]\n\n"
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
gStyle.SetPadLeftMargin(0.10)   # Left margin of the pads on the canvas.
gStyle.SetPadBottomMargin(0.10) # Bottom margin of the pads on the canvas.
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
num_canvases = 9
can = []
for i in range(0, num_canvases):
    name = "can%d" % (i) # Each canvas should have unique name
    title = "Data %d" % (i) # The title can be anything
    can.append(TCanvas( name, title, 10+10*i, 10+10*i, 1100, 600 ))
    can[i].SetFillColor( 0 )
    can[i].Divide( 1, 1 ) # Create 4 drawing pads in a 2x2 array.


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

###############################################################################
# Read each line of the file and grab some variables from that line. 
# Fill some histograms with this information.
###############################################################################
xpts = []
ypts = []
xerr = []
yerr = []
for i in range(0,6):
    xpts.append(array('d'))
    ypts.append(array('d'))
    xerr.append(array('d'))
    yerr.append(array('d'))

num_particles = 3
p4 = []
for i in range(0,num_particles):
    p4.append(TLorentzVector())

line = "First step"
while not line=='':

    good_event = True

    for i in range(0,num_particles):

        line = infile.readline()
        vars = line.split()

        print vars

        if len(vars)==2:
            if i == 0:
                xpts[0].append(float(vars[0]))
                ypts[0].append(float(vars[1]))
            elif i == 1:
                err = float(vars[1])
            elif i == 2:
                err = abs(err - float(vars[1]))/2.0

                xerr[0].append(0.0001)
                yerr[0].append(err)


qqbar_xsecs = [1.74, 0.35, 1.30]
qqbar_cumulative_xsecs = []
for i,xs in enumerate(qqbar_xsecs):
    if i==0:
        qqbar_cumulative_xsecs.append(xs)
    else:
        qqbar_cumulative_xsecs.append(xs+qqbar_cumulative_xsecs[i-1])


print xpts[0]
print ypts[0]
print xerr[0]
print yerr[0]

for i in range(1,4):
    for j in range(0,2500):

        xpts[i].append(9.0 + 0.001*j)
        ypts[i].append(qqbar_cumulative_xsecs[i-1])

        xerr[i].append(0.0001)
        yerr[i].append(0.005)

# Create an array of histograms
num_variables = 4

colors = [1,2,3,4]
gr = []
for i in range(0, num_variables):
    name = "gr_%d" % (i,) # Each histogram must have a unique name
    gr.append(TGraphErrors(len(xpts[i]),xpts[i],ypts[i],xerr[i],yerr[i]))
    gr[i].SetName(name)
    gr[i].SetTitle()

    # Set some formatting options
    gr[i].SetMinimum(0)

    gr[i].GetXaxis().SetNdivisions(8)
    gr[i].GetXaxis().SetLabelSize(0.06)
    #gr[i].GetXaxis().CenterTitle()
    gr[i].GetXaxis().SetTitleSize(0.06)
    gr[i].GetXaxis().SetTitleOffset(0.9)
    gr[i].GetXaxis().SetTitle("e^{+} e^{-} CM energy (GeV)")

    gr[i].GetYaxis().SetNdivisions(8)
    gr[i].GetYaxis().SetLabelSize(0.06)
    #gr[i].GetYaxis().CenterTitle()
    gr[i].GetYaxis().SetTitleSize(0.06)
    gr[i].GetYaxis().SetTitleOffset(0.8)
    gr[i].GetYaxis().SetTitle("Hadronic cross section (nb)")

    gr[i].SetMarkerStyle(20)
    gr[i].SetMarkerColor(colors[i])

    if i==0:
        gr[i].SetMarkerSize(1.0)
    else:
        gr[i].SetMarkerSize(1.5)

    gr[i].SetMinimum(0)
    gr[i].SetFillColor(colors[i])

legend = []
hdum = TH1F()
for i in range(0,num_canvases):
    legend.append(TLegend(0.85,0.65,0.99,0.99))
    if i>0:
        legend[i].AddEntry(gr[1],"u#bar{u}/d#bar{d}","f")
        if i==1:
            legend[i].AddEntry(hdum,"","")
            legend[i].AddEntry(hdum,"","")
            legend[i].AddEntry(hdum,"","")
    if i>1:
        legend[i].AddEntry(gr[2],"s#bar{s}","f")
        if i==2:
            legend[i].AddEntry(hdum,"","")
            legend[i].AddEntry(hdum,"","")
    if i>2:
        legend[i].AddEntry(gr[3],"c#bar{c}","f")
        if i==3:
            legend[i].AddEntry(hdum,"","")
    if i>3:
        legend[i].AddEntry(gr[0],"b#bar{b}","f")

################################################################################
# Draw the histograms 
################################################################################
i=0
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].SetLineColor(0)
gr[0].SetMarkerColor(0)
gr[0].GetXaxis().SetRangeUser(9.3,9.6)
gr[0].Draw("ap")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

i=1
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[1].SetMarkerColor(colors[1])
gr[1].GetXaxis().SetRangeUser(9.3,9.6)
gr[0].Draw("ap")
gr[1].Draw("p")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

i=2
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[1].GetXaxis().SetRangeUser(9.3,9.6)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

i=3
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[1].GetXaxis().SetRangeUser(9.3,9.6)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

i=4
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].SetMarkerColor(colors[0])
gr[0].SetLineColor(colors[0])
gr[0].GetXaxis().SetRangeUser(9.3,9.6)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
gr[0].Draw("p")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

i=5
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].GetXaxis().SetRangeUser(9.3,11.5)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
gr[0].Draw("p")
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

coords = [[9.5,16.5],
          [9.931,11.26],
          [10.2,8.1],
          [10.6,5.5]]
text = []
for j in range(0,4):
    text.append([])
    for i in range(0,4):

        x0 = coords[i][0]
        y0 = coords[i][1]
        x1 = x0+0.2
        y1 = y0+3.0

        text[j].append(TPaveText(x0,y0,x1,y1))
        name = "#varUpsilon (%dS)" % (i+1)
        text[j][i].AddText(name)
        text[j][i].SetFillColor(0)
        text[j][i].SetBorderSize(0)


i=6
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].GetXaxis().SetRangeUser(9.3,11.5)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
gr[0].Draw("p")
for t in text[0]:
    t.Draw()
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)


i=7
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].GetXaxis().SetRangeUser(9.9,10.7)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
gr[0].Draw("p")
for t in text[1]:
    t.Draw()
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)


ana_coords = [[9.90,19.0],
             [10.16,17.9],
             [10.4,20.4]]

ana_text = []
arrow = []
for j in range(0,3):
    ana_text.append([])
    arrow.append([])
    for i in range(0,3):

        x0 = ana_coords[i][0]
        y0 = ana_coords[i][1]
        x1 = x0+0.2
        y1 = y0+5.0

        ana_text[j].append(TPaveText(x0,y0,x1,y1))
        if i==0:
            ana_text[j][i].AddText("LFV")
            ana_text[j][i].AddText("#varUpsilon(nS)")
            arrow[j].append(TArrow(x0+0.05,y0-0.1,10.00,5.0,0.02,"|>"))
            arrow[j].append(TArrow(x0+0.05,y0-0.1,10.34,5.0,0.02,"|>"))

        elif i==1:
            ana_text[j][i].AddText("LFV/LNV")
            ana_text[j][i].AddText("Charm hadrons")
            arrow[j].append(TArrow(x0+0.05,y0-0.1,10.58,2.5,0.02,"|>"))
            arrow[j][2].SetLineColor(1)
            arrow[j][2].SetFillColor(5)

        elif i==2:
            ana_text[j][i].AddText("BNV/LNV")
            ana_text[j][i].AddText("B mesons")
            arrow[j].append(TArrow(x0+0.05,y0-0.1,10.58,5.9,0.02,"|>"))

        ana_text[j][i].SetTextColor(0)
        ana_text[j][i].SetFillColor(1)
        ana_text[j][i].SetBorderSize(0)



i=8
can[i].cd(1) # 
gPad.SetBottomMargin(0.12)
gPad.SetTopMargin(0.05)
gr[0].GetXaxis().SetRangeUser(9.9,10.7)
gr[0].Draw("ap")
gr[1].Draw("p")
gr[2].Draw("p")
gr[3].Draw("p")
gr[0].Draw("p")
for t in text[1]:
    t.Draw()
for t in ana_text[0]:
    t.Draw()
for t in arrow[0]:
    t.SetLineWidth(3)
    t.Draw()
legend[i].Draw()
gPad.Update()
name = "Plots/upsilon_xsec_%d.eps" % (i)
can[i].SaveAs(name)

###############################################################################
# Save the histograms as .ps files. 
# Note that this assumes the subdirectory "Plots" exists.
###############################################################################
'''
for i in range(0, num_canvases):
  name = "Plots/upsilon_xsec_%d.eps" % (i)
  can[i].SaveAs(name)
'''

###############################################################################
# Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
###############################################################################
if __name__ == '__main__':
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]
