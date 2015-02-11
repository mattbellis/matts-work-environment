#!/usr/bin/env python
                                                                                                                                        
                                                                                                                                        # import some modules
import sys
import ROOT
from ROOT import *
from optparse import OptionParser

from color_palette import *

from array import *

####################################################################

##################
# Use cool palette
##################
set_palette()

gStyle.SetFrameFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.20)

batchMode = False
###############################################
# Last argument determines batch mode or not
###############################################
numdiv = int(sys.argv[1])
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
###############################################
###############################################

can = []
for f in range(0, numdiv):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1200, 300))
  can[f].SetFillColor(0)
  can[f].Divide(1, 1)

########################################################
########################################################
divwidth = 0.98 / float(numdiv)
box = []
boxborder = []
textnum = []
for f in range(0, numdiv):
  box.append([])
  boxborder.append([])
  textnum.append([])
  for i in range(0, numdiv):
    can[f].cd(1)
    lox = 0.01 + i*divwidth
    loy = 0.01
    hix = 0.01 + (i+1)*divwidth 
    hiy = 0.99
    print str(f) + " " + str(i)
    print "\t" + str(lox) + " " + str(loy) + " " + str(hix) + " " + str(hiy)
    color = 2
    if f==i:
      color = 4
    box[f].append(TBox(lox, loy, hix, hiy))
    box[f][i].SetFillColor(color)
    box[f][i].Draw()

    boxborder[f].append(TBox(lox, loy, hix, hiy))
    boxborder[f][i].SetLineColor(1)
    boxborder[f][i].SetLineWidth(3)
    boxborder[f][i].SetFillStyle(0)
    boxborder[f][i].Draw("same")

    textnum[f].append(TPaveText(lox, loy, hix, hiy))
    textnum[f][i].SetFillStyle(0)
    textnum[f][i].SetTextSize(0.25)
    textnum[f][i].SetBorderSize(0)
    textnum[f][i].AddText(str(i+1))
    color = 1
    if f==i:
      color = 5
    textnum[f][i].SetTextColor(color)
    textnum[f][i].Draw()

    gPad.Update()

    name = "Plots/crossvalid_" + str(f) + "of" + str(numdiv) + ".eps"
    can[f].SaveAs(name)


########################################################
########################################################


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
                                                                                                                                                

