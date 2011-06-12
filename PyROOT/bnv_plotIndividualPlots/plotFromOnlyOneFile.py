#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import *

from color_palette import *

batchMode = False

nvars = 3
ncuts = 15
doTalkPlots = False
#
# Parse the command line options
#
htype = sys.argv[1]
tag = sys.argv[2]
ncuts = int(sys.argv[3])

###############################################
# Last argument determines batch mode or not
###############################################
last_file_offset = 0
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
  last_file_offset = -1
###############################################
###############################################

#############################
# Read in the filenames
#############################
filename = []
for f in range(4, len(sys.argv) + last_file_offset):
  print f
  filename.append(sys.argv[f])

if htype == "hmass":
  nvars = 3
elif htype == "hshape":
  nvars = 32
elif htype == "h2D":
  nvars = 2


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


canvastitles = ["B", "B","B"]

canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"
canvastitles[1] = "B^{0} #rightarrow #Lambda_{C}^{-} #mu^{+}"
canvastitles[2] = "B^{0} #rightarrow #Lambda_{C} #mu"

###############################
# Set up plots to display
###############################
plotstoshow = []
for i in range(0,32):
  plotstoshow.append([0])

plotstoshow[0] = [0]
plotstoshow[1] = [1]
plotstoshow[2] = [1,2]
plotstoshow[3] = [1,2,3]
plotstoshow[4] = [1,4,7]
plotstoshow[5] = [2,5,8]
plotstoshow[6] = [3,6,9]
plotstoshow[7] = [2]
plotstoshow[8] = [3]

canvastext = []
can = []
toppad = []
bottompad = []
legend = []
for f in range(0,nvars*ncuts):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 600, 400 ))
  #can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 600, 400 ))
  #can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 600, 200 ))
  #can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 200, 150 ))
  can[f].SetFillColor( 0 )
  can[f].SetFillStyle(1001)
  #can[f].Divide( 5,6 )
  name = "top" + str(f)
  #toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  toppad.append(TPad(name, name, 0.01, 0.98, 0.99, 0.99))
  toppad[f].SetFillColor(0)
  toppad[f].SetFillStyle(1001)
  toppad[f].Draw()
  toppad[f].Divide(1,1)
  name = "bottom" + str(f)
  #bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.99))
  bottompad[f].SetFillColor(0);
  bottompad[f].SetFillStyle(1001);
  bottompad[f].Draw();
  bottompad[f].Divide(1, 1);

  can[f].Update()

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(canvastitles[f])
  canvastext[f].AddText("Stuff")
  canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  #canvastext[f].Draw()


hmax = []
histos = []
text1 = []
for f in range(0,8):
  histos.append([])
  text1.append([])
  for i in range(0,1):
    histos[f].append([])
    text1[f].append([])
    hmax.append([])
    for j in range(0,32):
      histos[f][i].append([])
      text1[f][i].append([])
      hmax[i].append([])
      for k in range(0,16):
        histos[f][i][j].append([])
        text1[f][i][j].append([])
        hmax[i][j].append([])

datasets = ["SP9446", "OffPeak", "OnPeak", "SP1005", "SP998", "SP1235", "SP1237"]
datasettext = ["Signal MC", "Off peak", "On Peak (blind)", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic", "Generic MC (All)"] 
colors = [22, 6, 4, 23, 26, 30, 36, 4, 6, 6]
scale_amount = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

hlabel = [datasettext[0], datasettext[7], datasettext[2]]

#
# Open a ROOT file and save the formula, function and histogram
#
numfiles = len(filename)
rootfile = []
for f in range(0,numfiles):
  print filename[f]
  rootfile.append(TFile( filename[f] ))
  if os.path.isfile( filename[f] ):
    #print f
    print filename[f]
    for i in range(0, 1):
      for j in range(0,nvars):
        for k in range(0,ncuts):
          if(1):
            hname = htype + str(i) + "_" + str(j) + "_" + str(k)
            #print hname
            histos[f][i][j][k] = gROOT.FindObject(hname)
            if histos[f][i][j][k]:
              newname = htype+str(f)+"_" + str(i) + "_" + str(j) + "_" + str(k)
              histos[f][i][j][k].SetName(newname)
            else:
              if htype != "h2D":
                histos[f][i][j][k] = TH1F(hname,hname,10,0,1)
              else:
                histos[f][i][j][k] = TH2F(hname,hname,10,0,1, 10, 0,1)

            hdum = histos[f][i][j][k].Clone()
          # Draw the canvas labels
          
          bottompad[k + ncuts*j].cd(1)

          #histos[f][i][j][k].Sumw2()

          #if(j==2):
            #histos[f][i][j][k].Rebin(3)
          #else:
          #if htype != "h2D":
            #histos[f][i][j][k].Rebin(2)

          #print str(f) + str(i) + str(j) + " " + str(k) + " " + str(histos[0][i][j][k].GetEntries()) + " " + str(histos[f][i][j][k].GetEntries())
          print str(f) + str(i) + str(j) + " " + str(k) + " entries: " + str(histos[f][i][j][k].GetEntries()) + "\tintegral: " + str(histos[f][i][j][k].Integral())
          histos[f][i][j][k].SetMinimum(0)
          histos[f][i][j][k].SetTitle("")
          
          histos[f][i][j][k].GetYaxis().SetNdivisions(4)
          histos[f][i][j][k].GetXaxis().SetNdivisions(6)
          histos[f][i][j][k].GetYaxis().SetLabelSize(0.06)
          histos[f][i][j][k].GetXaxis().SetLabelSize(0.06)

          histos[f][i][j][k].GetXaxis().CenterTitle()
          histos[f][i][j][k].GetXaxis().SetTitleSize(0.09)
          histos[f][i][j][k].GetXaxis().SetTitleOffset(1.0)
          #histos[f][i][j][k].GetXaxis().SetTitle(xaxistitle)

          #####################
          # Get the maximum for a given set of cuts
          #####################
          if f==0:
            hmax[i][j][k] = histos[f][i][j][k].GetMaximum()
          else:
            if hmax[i][j][k] < histos[f][i][j][k].GetMaximum():
              hmax[i][j][k] = histos[f][i][j][k].GetMaximum()

          histos[f][i][j][k].SetLineColor(1)
          histos[f][i][j][k].SetLineWidth(3)

          if f>0:
            histos[f][i][j][k].SetFillStyle(3004 + f-1)
            histos[f][i][j][k].SetLineWidth(4)
            histos[f][i][j][k].SetLineColor(colors[f])

          histos[f][i][j][k].SetFillColor(colors[k])
          
          ###########################

###################################
# Draw the plots
###################################
for f in range(0,numfiles):
  for i in range(0, 1):
    legend.append([])
    for j in range(0,nvars):
      legend[i].append([])
      for k in range(0,ncuts):
        bottompad[k + ncuts*j].cd(1)
        if f==0:
          legend[i][j].append(TLegend(0.60,0.90,0.99,0.99))
        #legend[i][j][k].AddEntry( histos[f][i][j][k], hlabel[f], "f")
        legend[i][j][k].AddEntry( histos[f][i][j][k], hlabel[f], "l")
        option = ""
        if htype != "h2D":
          #print "hmax: " + str(i) + " " + str(j) + " " + str(k) + " " + str(hmax[i][j][k]) 
          histos[f][i][j][k].SetMaximum( 1.1*hmax[i][j][k] )

        else:
          if histos[f][i][j][k].GetListOfFunctions().FindObject("palette"):
            histos[f][i][j][k].GetListOfFunctions().FindObject("palette").SetX1NDC(0.90)
            histos[f][i][j][k].GetListOfFunctions().FindObject("palette").SetX2NDC(0.95)
            histos[f][i][j][k].GetListOfFunctions().FindObject("palette").SetY1NDC(0.20)
            histos[f][i][j][k].GetListOfFunctions().FindObject("palette").SetY2NDC(0.90)
          option = "col"

        ###############################################
        #  Draw what we need 
        ###############################################
        for count,n in enumerate(plotstoshow[k]):
          print n
          print count
          if count==0:
            histos[f][i][j][n].DrawCopy(option)
          else:
            histos[f][i][j][n].DrawCopy(option + "same")

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

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
