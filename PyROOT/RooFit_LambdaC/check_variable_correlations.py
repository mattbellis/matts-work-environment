#!/usr/bin/env python

import sys
from optparse import OptionParser

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option('-t', "--tag", dest="tag", default='default', help="Tag to append to output files")
parser.add_option("--batch", dest="batch", default=False, action='store_true', help="Run in batch mode.")

(options, args) = parser.parse_args()

infile = None
if len(args[0])>0:
  infile = open(args[0])
else:
  print "No file!"
  exit(-1)
#######################################################
#######################################################

import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

#######################################################
#######################################################
gROOT.Reset()
gStyle.SetOptStat(0)
gStyle.SetOptFit(0)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
set_palette("palette",100)
# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetTitleYOffset(2.00)

#######################################################
#######################################################
#######################################################

nplots = 16
h = []
h2D = []
for i in range(0,3): # Types of plots
  h.append([])
  h2D.append([])
  for j in range(0,6):
    h[i].append([])
    h2D[i].append([])
    for k in range(0,6):
      name = "h%d_%d_%d" % (i, j, k)
      h[i][j].append(TH1F(name, name, 20, 0.75, 1.0))
      h[i][j][k].SetLineColor(2)
      h[i][j][k].SetLineWidth(4)
      h[i][j][k].SetFillStyle(1001)
      h[i][j][k].SetFillColor(4)

      h[i][j][k].SetMinimum(0)
      if i==1:
        h[i][j][k].SetMaximum(1.0)
        h[i][j][k].SetMinimum(-1.0)
        h[i][j][k].GetYaxis().SetRangeUser(-0.01, 0.01)

      h[i][j][k].SetTitle("")

      h[i][j][k].GetXaxis().SetNdivisions(6)
      h[i][j][k].GetXaxis().SetLabelSize(0.06)
      h[i][j][k].GetXaxis().CenterTitle()
      h[i][j][k].GetXaxis().SetTitleSize(0.09)
      h[i][j][k].GetXaxis().SetTitleOffset(0.8)
      h[i][j][k].GetXaxis().SetTitle( "MLP output" )

      h[i][j][k].GetYaxis().SetNdivisions(4)
      h[i][j][k].GetYaxis().SetLabelSize(0.06)
      h[i][j][k].GetYaxis().CenterTitle()
      h[i][j][k].GetYaxis().SetTitleSize(0.09)
      h[i][j][k].GetYaxis().SetTitleOffset(1.0)
      if i==0:
        h[i][j][k].GetYaxis().SetTitle("# events/bin")
      elif i==1:
        h[i][j][k].GetYaxis().SetTitle("Ratio")



      name = "h2D%d_%d_%d" % (i, j, k)
      h2D[i][j].append(TH2F(name, name, 20, 5.2, 5.3, 20, -0.2, 0.2))
      h2D[i][j][k].SetTitle("")

      h2D[i][j][k].GetXaxis().SetNdivisions(6)
      h2D[i][j][k].GetXaxis().SetLabelSize(0.06)
      h2D[i][j][k].GetXaxis().CenterTitle()
      h2D[i][j][k].GetXaxis().SetTitleSize(0.09)
      h2D[i][j][k].GetXaxis().SetTitleOffset(0.8)
      h2D[i][j][k].GetXaxis().SetTitle( "m_{ES}" )

      h2D[i][j][k].GetYaxis().SetNdivisions(4)
      h2D[i][j][k].GetYaxis().SetLabelSize(0.06)
      h2D[i][j][k].GetYaxis().CenterTitle()
      h2D[i][j][k].GetYaxis().SetTitleSize(0.09)
      h2D[i][j][k].GetYaxis().SetTitleOffset(1.0)
      h2D[i][j][k].GetYaxis().SetTitle( "#Delta E" )
   

#######################################################
# Read in values
#######################################################
for line in infile:
  vals = line.split()
  mes = float(vals[0])
  dle = float(vals[1])
  mlp = float(vals[2])
  x = int( 10.0*(mes-5.2)/0.25 ) + 1
  y = int( 10.0*(0.2-dle)/1.0 ) + 1
  print "%d %d %f %f" % ( x, y, mes, dle )
  h[0][x][y].Fill(mlp)

  h2D[0][x][y].Fill(mes, dle)

  # Everything
  h2D[0][0][0].Fill(mes, dle)
  h[0][0][0].Fill(mlp)

  if dle<0.075 and dle>-0.100 and mes>5.270:
    h[0][4][5].Fill(mlp)
    h2D[0][4][5].Fill(mes, dle)
  else:
    h[0][5][5].Fill(mlp)
    h2D[0][5][5].Fill(mes, dle)

#######################################################
#######################################################


#######################################################
#######################################################
can = []
for i in range(0,5):
  name = "can%d" % (i)
  can.append(TCanvas(name, name, 10+10*i, 10+10*i, 1200, 600))
  can[i].SetFillColor(0)
  if i<4:
    can[i].Divide(5,5)
  else:
    can[i].Divide(3,2)


#######################################################
#######################################################
for j in range(0,5):
  for k in range(0,5):
    ipad = 5*k + j + 1
    can[0].cd(ipad)
    h2D[0][j][k].Draw()
    gPad.Update()

    can[1].cd(ipad)
    h[0][j][k].Draw("e")
    gPad.Update()

    can[2].cd(ipad)
    h[1][j][k].Sumw2()
    if h[0][j][k].GetEntries()!=0.0 and h[0][2][2].GetEntries()!=0.0:
      h[1][j][k].Add(h[0][j][k], h[0][2][2], 1.0/h[0][j][k].Integral(), -1.0/h[0][2][2].Integral())
      h[1][j][k].Divide(h[0][2][2])
      #h[1][j][k].Divide(h[0][j][k], h[0][2][2], 1.0/h[0][j][k].Integral(), 1.0/h[0][2][2].Integral())
      #h[1][j][k].SetMinimum(0.0)
      #h[1][j][k].SetMaximum(3.0)
      h[1][j][k].Fit('1.0++x')
      h[1][j][k].GetYaxis().SetRangeUser(-0.01, 0.01)
      h[1][j][k].Draw('e')
      gPad.Update()

    can[3].cd(ipad)
    h[2][j][k].Sumw2()
    if h[0][j][k].GetEntries()!=0.0 and h[0][0][0].GetEntries()!=0.0:
      h[2][j][k].Add(h[0][j][k], h[0][0][0], 1.0/h[0][j][k].Integral(), -1.0/h[0][0][0].Integral())
      h[2][j][k].Divide(h[0][0][0])
      #h[2][j][k].Divide(h[0][j][k], h[0][2][2], 1.0/h[0][j][k].Integral(), 1.0/h[0][2][2].Integral())
      #h[2][j][k].SetMinimum(0.0)
      #h[2][j][k].SetMaximum(3.0)
      h[2][j][k].Fit('1.0++x')
      h[2][j][k].GetYaxis().SetRangeUser(-0.001, 0.001)
      h[2][j][k].Draw('e')
      gPad.Update()

#####################################################
# Signal area plot
#####################################################
sigx = 4
sigy = 5
refx = 5
refy = 5
can[4].cd(1)
h2D[0][sigx][sigy].Draw()
gPad.Update()

can[4].cd(2)
h[0][sigx][sigy].Draw('e')
gPad.Update()

can[4].cd(3)
if h[0][sigx][sigy].GetEntries()!=0.0 and h[0][refx][refy].GetEntries()!=0.0:
  h[1][sigx][sigy].Sumw2()
  h[1][sigx][sigy].Add(h[0][sigx][sigy], h[0][refx][refy], 1.0/h[0][sigx][sigy].Integral(), -1.0/h[0][refx][refy].Integral())
  h[1][sigx][sigy].Divide(h[0][refx][refy])
  #h[1][sigx][sigy].Divide(h[0][sigx][sigy], h[0][refx][refy], 1.0/h[0][sigx][sigy].Integral(), 1.0/h[0][refx][refy].Integral())
  h[1][sigx][sigy].Draw('e')
  h[1][sigx][sigy].Fit('1.0++x')
  h[1][sigx][sigy].GetYaxis().SetRangeUser(-0.01, 0.01)
  h[1][sigx][sigy].Draw('e')
  gPad.Update()

can[4].cd(4)
h2D[0][refx][refy].Draw()
gPad.Update()

can[4].cd(5)
h[0][refx][refy].Draw('e')
gPad.Update()

can[4].cd(6)
if h[0][refx][refy].GetEntries()!=0.0:
  h[1][refx][refy].Sumw2()
  h[1][refx][refy].Add(h[0][refx][refy], h[0][refx][refy], 1.0/h[0][refx][refy].Integral(), -1.0/h[0][refx][refy].Integral())
  h[1][refx][refy].Divide(h[0][refx][refy])
  #h[1][refx][refy].Divide(h[0][refx][refy], h[0][refx][refy], 1.0/h[0][refx][refy].Integral(), 1.0/h[0][refx][refy].Integral())
  h[1][refx][refy].GetYaxis().SetRangeUser(-0.01, 0.01)
  h[1][refx][refy].Draw("e")
  gPad.Update()

#######################################################
#######################################################
for i in range(0,4):
  name = "Plots_correlations/can_%s_%d.eps" % (options.tag, i)
  can[i].SaveAs(name)
#######################################################
#######################################################

#######################################################
#######################################################
## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


