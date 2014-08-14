#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import *

from color_palette import *

batchMode = False

#
# Parse the command line options
#
#filename = ["default.root", "default.root", "default.root", ]
#filename[1] = sys.argv[2]
#filename[2] = sys.argv[3]
#whichType = sys.argv[4]

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
gStyle.SetPadBottomMargin(0.00)
gStyle.SetPadTopMargin(0.00)
gStyle.SetPadRightMargin(0.00)
gStyle.SetPadLeftMargin(0.00)
gStyle.SetFrameFillStyle(0)
gStyle.SetFrameFillColor(0)
set_palette("palette",100)

canvastitles = ["B"]
canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,4):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 600, 600 ))
  can[f].SetFillColor( 0 )
  can[f].Divide( 1, 1, 0, 0 )
  #name = "top" + str(f)
  #toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  #toppad[f].SetFillColor(0)
  #toppad[f].Draw()
  #toppad[f].Divide(2,2)
  #name = "bottom" + str(f)
  #bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  #bottompad[f].SetFillColor(0);
  #bottompad[f].Draw();
  #bottompad[f].Divide(2, 2);

  #toppad[f].cd(1)
  #canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(canvastitles[f])
  #canvastext[f].AddText("")
  #canvastext[f].SetBorderSize(1)
  #canvastext[f].SetFillStyle(1)
  #canvastext[f].SetFillColor(1)
  #canvastext[f].SetTextColor(0)
  #canvastext[f].Draw()


x0 = 0.5
y0 = 0.5
r0 = 0.3
r1 = 0.40
e0_blank = []
e0 = []
e0_fill = []
ewedge = []
for f in range(0,4):
  can[f].cd(1)
  if f%2==0:
    e0.append(TEllipse(x0, x0, r0, r0, 60.0, 300.0)) 
  elif f%2==1:
    e0.append(TEllipse(x0, x0, r0, r0, -60.0, 60.0)) 
  e0[f].SetFillStyle(1001)
  e0[f].SetFillColor(1)
  e0[f].Draw()

  if f%2==0:
    e0_fill.append(TEllipse(x0, x0, r0, r0, -60.0, 60.0)) 
  elif f%2==1:
    e0_fill.append(TEllipse(x0, x0, r0, r0, 60.0, 300.0)) 
  e0_fill[f].SetFillColor(0)
  e0_fill[f].Draw()

  e0_blank.append(TEllipse(x0, x0, r0, r0, 0.0, 360.0)) 
  e0_blank[f].SetLineWidth(20)
  e0_blank[f].SetLineColor(22)
  e0_blank[f].SetFillColor(0)
  e0_blank[f].Draw()

  ewedge.append([])
  for w in range(0,3):
    if f<2:
      ewedge[f].append(TEllipse(x0, x0, r1, r1, 0.0 + 120*w, 115.0 + 120*w)) 
      ewedge[f][w].SetLineColor(1)
      ewedge[f][w].SetLineWidth(20)
      ewedge[f][w].SetFillColor(2+w)
    else:
      ewedge[f].append(TEllipse(x0, x0, r1, r1, 0.0 + 120*w, 115.0 + 120*w)) 
      ewedge[f][w].SetLineColor(2+w)
      ewedge[f][w].SetLineWidth(20)
      ewedge[f][w].SetFillColor(1)

    ewedge[f][w].Draw()

  e0_blank[f].Draw()
  e0[f].Draw()
  e0_fill[f].Draw()

  gPad.Update()





for j in range(0,2):
  name = "Plots/can" + str(j) + ".ps" 
  can[j].SaveAs(name)
  name = "Plots/can" + str(j) + ".eps" 
  can[j].SaveAs(name)
  name = "Plots/can" + str(j) + ".png" 
  can[j].SaveAs(name)
  name = "Plots/can" + str(j) + ".pdf" 
  can[j].SaveAs(name)
  name = "Plots/can" + str(j) + ".jpg" 
  can[j].SaveAs(name)



## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
