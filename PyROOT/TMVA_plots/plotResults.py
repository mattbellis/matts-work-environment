#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import gROOT, gStyle

from color_palette import *

batchMode = False

#
# Parse the command line options
#
filename = ["default.root", "default.root", "default.root", ]
filename[0] = sys.argv[1]
filename[1] = sys.argv[2]
filename[2] = sys.argv[3]
whichType = sys.argv[4]

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
gStyle.SetOptStat(10)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.3);                
gStyle.SetStatW(0.25);                
gStyle.SetPadBottomMargin(0.20)
set_palette("palette",100)

canvastitles = ["B"]
canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,1):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1400, 900 ))
  can[f].SetFillColor( 0 )
  #can[f].Divide( 5,6 )
  name = "top" + str(f)
  toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  toppad[f].SetFillColor(0)
  toppad[f].Draw()
  toppad[f].Divide(2,2)
  name = "bottom" + str(f)
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  bottompad[f].SetFillColor(0);
  bottompad[f].Draw();
  bottompad[f].Divide(2, 2);

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  canvastext[f].AddText(canvastitles[f])
  canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()



numfiles = 3
histos = []
text1 = []
for f in range(0,numfiles):
  histos.append([])
  text1.append([])
  for i in range(0,3):
    histos[f].append([])
    text1[f].append([])

datasettext = ["Data"]
xaxistitle = "Mass GeV/c^{2}"

#
# Open a ROOT file and save the formula, function and histogram
#
for f in range(0,numfiles):
  print filename[f]
  rootfile = TFile( filename[f] )
  if os.path.isfile(filename[f] ):
    rootfile.ls()
    #print f
    print filename[f]
    for i in range(0,3):
      hname = whichType + str(i)
      print hname
      histos[f][i] = gROOT.FindObject(hname)
      if histos[f][i]:
        newname = whichType+str(f)+"_" + str(i) 
        histos[f][i].SetName(newname)
        print newname
      else:
        histos[f][i] = TH1F(hname,hname,10,0,1)


      # Draw the canvas labels
      bottompad[0].cd(f+1)
      histos[f][i].SetMinimum(0)
      histos[f][i].SetTitle("")
      
      histos[f][i].GetYaxis().SetNdivisions(4)
      histos[f][i].GetXaxis().SetNdivisions(6)
      histos[f][i].GetYaxis().SetLabelSize(0.06)
      histos[f][i].GetXaxis().SetLabelSize(0.06)

      histos[f][i].GetXaxis().CenterTitle()
      histos[f][i].GetXaxis().SetTitleSize(0.09)
      histos[f][i].GetXaxis().SetTitleOffset(1.0)
      histos[f][i].GetXaxis().SetTitle(xaxistitle)

      if(i==0):
        histos[f][i].SetFillColor(22)
      elif(i==1):
        histos[f][i].SetFillColor(0)
        histos[f][i].SetLineColor(2)
        histos[f][i].SetLineWidth(6)
      elif(i==2):
        histos[f][i].SetFillColor(0)
        histos[f][i].SetLineColor(4)
        histos[f][i].SetLineWidth(6)

      if(i==0):
        histos[f][i].DrawCopy()
      else:
        histos[f][i].DrawCopy("same")
          
      bottompad[0].cd(f + 1)
      text1[f][i] = TPaveText(0.0, 0.9, 0.4, 1.0, "NDC")
      text1[f][i].AddText(filename[f])
      text1[f][i].SetBorderSize(1)
      text1[f][i].SetFillStyle(1)
      text1[f][i].Draw()

    can[0].Update()



for j in range(0,1):
  name = "plots/can" + str(j) + "_" + whichType + ".ps" 
  can[j].SaveAs(name)

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
