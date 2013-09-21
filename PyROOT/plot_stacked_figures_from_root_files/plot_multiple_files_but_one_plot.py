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
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)


can = []
legend = []
for f in range(0,1):
    name = "can" + str(f)
    can.append(TCanvas( name, name, 10+10*f, 10+10*f,600,600))
    can[f].SetFillColor( 0 )
    can[f].SetFillStyle(1001)
    can[f].Divide(1,1)

    can[f].Update()

colors = [22, 6, 4, 23, 26, 30, 36, 4, 6]

hmax = 0.0
histos = []
numfiles = len(filename)
rootfile = []
for f in range(0,numfiles):
    print filename[f]
    rootfile.append(TFile( filename[f] ))
    if os.path.isfile( filename[f] ):
        #print f
        print filename[f]
        histos.append(gROOT.FindObject(hname))
        if histos[f]:
            newname = "%s_%d" % (hname,f)
            histos[f].SetName(newname)
        else:
            histos[f] = TH2F(hname,hname,10,0,1, 10, 0,1)

        #scale_amount[f] = float(histos[0].Integral())/ histos[f].Integral() 
        #print "scale: " + str(scale_amount[f])
        #histos[f].Scale(scale_amount[f])
        histos[f].SetMinimum(0)
        histos[f].SetTitle("")

        histos[f].GetYaxis().SetNdivisions(4)
        histos[f].GetXaxis().SetNdivisions(6)
        histos[f].GetYaxis().SetLabelSize(0.06)
        histos[f].GetXaxis().SetLabelSize(0.06)

        histos[f].GetXaxis().CenterTitle()
        histos[f].GetXaxis().SetTitleSize(0.09)
        histos[f].GetXaxis().SetTitleOffset(1.0)
        #histos[f].GetXaxis().SetTitle(xaxistitle)

        #####################
        # Get the maximum for a given set of cuts
        #####################
        if f==0:
            hmax = histos[f].GetMaximum()
        else:
            if hmax < histos[f].GetMaximum():
                hmax = histos[f].GetMaximum()

        histos[f].SetLineColor(colors[f])
        histos[f].SetLineWidth(3)

        if f>0:
            histos[f].SetFillStyle(3004 + f-1)

        histos[f].SetLineWidth(4)
        #histos[f].SetLineColor(colors[f])

        histos[f].SetFillColor(colors[f])


for f in range(0,numfiles):
      can[0].cd(1)
      if f==0:
          histos[f].SetMaximum(hmax)
          histos[f].Draw()
          histos[f].Draw("samee")
      else:
          histos[f].Draw("samee")
      gPad.Update()

        
      ###########################

###################################
# Draw the plots
###################################
'''
for f in range(0,numfiles):
  for i in range(0, 1):
    legend.append([])
    for j in range(0,nvars):
      legend[i].append([])
      for k in range(0,ncuts):
        bottompad[k + ncuts*j].cd(1)
        if f==0:
          legend[i][j].append(TLegend(0.70,0.80,0.99,0.99))
        legend[i][j][k].AddEntry( histos[f], hlabel[f], "f")
        if htype != "h2D":
          #print "hmax: " + str(i) + " " + str(j) + " " + str(k) + " " + str(hmax[i][j][k]) 
          histos[f].SetMaximum( 1.1*hmax[i][j][k] )
          if f==0:
            histos[f].DrawCopy()
          else:
            histos[f].DrawCopy("same")
        else:
          histos[f].GetListOfFunctions().FindObject("palette").SetX1NDC(0.90)
          histos[f].GetListOfFunctions().FindObject("palette").SetX2NDC(0.95)
          histos[f].GetListOfFunctions().FindObject("palette").SetY1NDC(0.20)
          histos[f].GetListOfFunctions().FindObject("palette").SetY2NDC(0.90)
          histos[f].DrawCopy("colz")

        legend[i][j][k].SetFillColor(0)
        legend[i][j][k].Draw()
        bottompad[k + ncuts*j].Update()


        #bottompad[k + ncuts*j].cd(f + 1)
        #text1[f][i][j][k] = TPaveText(0.0,0.9,0.4,1.0,"NDC")
        #text1[f][i][j][k].AddText(datasettext[f])
        #text1[f][i][j][k].SetBorderSize(1)
        #text1[f][i][j][k].SetFillStyle(1)
        #text1[f][i][j][k].Draw()


###############################
# Save the canvases
###############################
for j in range(0,nvars):
  for k in range(0,ncuts):
    can[k+ncuts*j].Update()
    name = "Plots/" + htype + "_" + str(j) + "_" + str(k) + "_" + tag + ".eps"
    can[k+ncuts*j].SaveAs(name)
'''

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
