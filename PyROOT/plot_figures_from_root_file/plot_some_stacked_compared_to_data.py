#!/usr/bin/env python2.5
#
#

# Import the needed modules
import os
import sys

from optparse import OptionParser

from ROOT import *

from color_palette import *

batchMode = False

################################################################################
parser = OptionParser()
parser.add_option("--hname", dest="hname", default="hmass0_0_0", \
    help="Histogram to plot.")
parser.add_option("--tag", dest="tag", default="default", \
    help="Tag to add on to output files.")
parser.add_option("--batch", dest="batch", default=False, action="store_true", \
    help="Run in batch mode")

(options, args) = parser.parse_args()

#
# Parse the command line options
#
hname = options.hname
tag = options.tag

################################################################################
#############################
# Read in the filenames
#############################
filename = args


gROOT.Reset()
#gStyle.SetOptStat(10)
gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.10)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)


can = []
legend = []
for f in range(0,1):
    name = "can" + str(f)
    can.append(TCanvas( name, name, 10+10*f, 10+10*f,700,400))
    can[f].SetFillColor( 0 )
    can[f].SetFillStyle(1001)
    can[f].Divide(1,1)

    can[f].Update()

colors = [22, 6, 4, 23, 26, 30, 36, 4, 6]

hmax = 0.0
histos = []
numfiles = len(filename)
rootfile = []

scale_factor = 1.0

# Create the stacked histo
histos.append(THStack("hstack","Stacked histogram"))

################################################################################
hcount = 0
for f in range(0,numfiles):

    scale_factor = 1.0
    if filename[f].find('genericQQbar')>=0:
        scale_factor = 0.5
    elif filename[f].find('genericBBbar')>=0:
        scale_factor = 0.33

    print filename[f]
    rootfile.append(TFile( filename[f] ))
    if os.path.isfile( filename[f] ):
        #print f
        #print filename[f]
        hdum = gROOT.FindObject(hname)
        if not hdum:
            hdum = TH1F("hdum","hdum",100,0,100)
        hdum.Scale(scale_factor)
        #print hdum
        if hname.find('_0_')<0:
            hdum.Rebin(4)
        else:
            hdum.Rebin(2)

        if hdum:
          hdum.SetMinimum(0)
          hdum.SetTitle("")

          hdum.GetYaxis().SetNdivisions(4)
          hdum.GetYaxis().SetLabelSize(0.06)
          hdum.GetYaxis().SetTitleSize(0.07)
          hdum.GetYaxis().SetTitleOffset(0.7)

          hdum.GetXaxis().SetNdivisions(6)
          hdum.GetXaxis().SetLabelSize(0.06)

          hdum.GetXaxis().CenterTitle()
          hdum.GetXaxis().SetTitleSize(0.09)
          hdum.GetXaxis().SetTitleOffset(1.0)
          #hdum.GetXaxis().SetTitle(xaxistitle)

          #####################
          # Get the maximum for a given set of cuts
          #####################
          '''
          if f==0:
              hmax = hdum.GetMaximum()
          else:
              if hmax < hdum.GetMaximum():
                  hmax = hdum.GetMaximum()
          '''

          #hdum.SetLineColor(colors[f])
          hdum.SetLineWidth(3)

          # Set the x axis
          if hname.find('_0_')<0:
              hdum.GetXaxis().SetRangeUser(5.2,5.3)

          '''
          if f>0:
              hdum.SetFillStyle(3004 + f-1)
          '''

          hdum.SetLineWidth(4)
          #hdum.SetLineColor(colors[hcount])

          #hdum.SetFillColor(colors[hcount])

        if not f==numfiles-1:
            hdum.SetFillColor(colors[f])
            histos[0].Add(hdum)
            hcount = 0
        else:
            hdum.SetFillColor(0)
            histos.append(hdum)
            hcount = 1
            newname = "%s_%d" % (hname,hcount)
            #print newname
            histos[hcount].SetName(newname)

# Set the scaling
hmax0 = histos[0].GetMaximum()
hmax1 = histos[1].GetMaximum()
hmax = hmax0
if hmax1>hmax:
    hmax = hmax1
#print hmax

can[0].cd(1)
histos[0].SetMaximum(1.2*hmax)
histos[1].SetMaximum(1.2*hmax)
histos[1].Draw("e")
histos[0].Draw("same")
#histos[0].Draw("samee")
histos[1].Draw("samee")
gPad.Update()


###########################

###################################
# Draw the plots
###################################
legend = TLegend(0.75,0.70,0.99,0.99)
hdum0 = TH1F("h0","h",10,0,10)
hdum0.SetLineWidth(6)
legend.AddEntry( hdum0, "Sideband data", "le")

hdum1 = TH1F("h1","h",10,0,10)
hdum1.SetFillColor(colors[0])
legend.AddEntry( hdum1, "B#bar{B} MC", "f")

hdum2 = TH1F("h2","h",10,0,10)
hdum2.SetFillColor(colors[1])
legend.AddEntry( hdum2, "q#bar{q} MC", "f")

legend.SetFillColor(0)
legend.Draw()
gPad.Update()



#'''
###############################
# Save the canvases
###############################
for c in can:
    gPad.Update()
    name = "Plots/%s_%s%s.eps" % (c.GetName(),hname,options.tag)
    c.SaveAs(name)
#'''

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
