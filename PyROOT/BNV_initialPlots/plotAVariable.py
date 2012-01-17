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
baryon = sys.argv[1]
ntp = sys.argv[2]
whichType = sys.argv[3]
whichPlot = int(sys.argv[4])

drange = False
if baryon.find("Drange") > -1:
  drange = True

print "drange: " + str(drange)

index = []
if len(sys.argv)>=7:
  index = [ int(sys.argv[5]), int(sys.argv[6]) ]

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

#mode = ["1235", "1237", "1005", "998"]
mode = ["9445", "9446", "1235", "1237", "1005", "998"]
#mode = ["9445_unblind__conLambdaC", "9446_unblind__conLambdaC", "1235", "1237", "1005", "998"]
if baryon == "Lambda0":
  if ntp=="1" or ntp=="3":
    mode = ["9452", "9454", "1235", "1237", "1005", "998"]
  elif ntp=="2" or ntp=="4":
    mode = ["9453", "9455", "1235", "1237", "1005", "998"]
elif baryon == "proton":
  mode = ["9456", "9457", "1235", "1237", "1005", "998"]


gROOT.Reset()
#gStyle.SetOptStat(10)
gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadBottomMargin(0.20)
#gStyle.SetPalette(1)
set_palette("palette",100)


canvastitles = ["B", "B","B"]

if baryon == "LambdaC":
  if ntp == "1":
    canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"
    canvastitles[1] = "B^{0} #rightarrow #Lambda_{C}^{-} #mu^{+}"
    canvastitles[2] = "B^{0} #rightarrow #Lambda_{C} #mu"
  if ntp == "2":
    canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} e^{-}"
    canvastitles[1] = "B^{0} #rightarrow #Lambda_{C}^{-} e^{+}"
    canvastitles[2] = "B^{0} #rightarrow #Lambda_{C} e"
elif baryon == "Lambda0":
  if ntp == "1":
    canvastitles[0] = "B^{-} #rightarrow #Lambda_{S}^{0} #mu^{-}"
    canvastitles[1] = "B^{+} #rightarrow #bar{#Lambda_{S}^{0}} #mu^{+}"
    canvastitles[2] = "B #rightarrow #Lambda_{S}^{0} #mu"
  if ntp == "2":
    canvastitles[0] = "B^{-} #rightarrow #bar{#Lambda_{S}^{0}} #mu^{-}"
    canvastitles[1] = "B^{+} #rightarrow #Lambda_{S}^{0} #mu^{+}"
    canvastitles[2] = "B #rightarrow #Lambda_{S}^{0} #mu"
  if ntp == "3":
    canvastitles[0] = "B^{-} #rightarrow #bar{#Lambda_{S}^{0}} e^{-}"
    canvastitles[1] = "B^{+} #rightarrow #Lambda_{S}^{0} e^{+}"
    canvastitles[2] = "B #rightarrow #Lambda_{S}^{0} e"
  if ntp == "4":
    canvastitles[0] = "B^{-} #rightarrow #bar{#Lambda_{S}^{0}} e^{-}"
    canvastitles[1] = "B^{+} #rightarrow #Lambda_{S}^{0} e^{+}"
    canvastitles[2] = "B #rightarrow #Lambda_{S}^{0} e"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,3):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1400, 900 ))
  can[f].SetFillColor( 0 )
  #can[f].Divide( 5,6 )
  name = "top" + str(f)
  toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  toppad[f].SetFillColor(0)
  toppad[f].Draw()
  toppad[f].Divide(1,1)
  name = "bottom" + str(f)
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  bottompad[f].SetFillColor(0);
  bottompad[f].Draw();
  bottompad[f].Divide(8, 6);

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  canvastext[f].AddText(canvastitles[f])
  canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()


#
# Single canvases
#
cansingle = []
if len(index)==2:
  for f in range(0,8):
    name = "cansingle" + str(f)
    cansingle.append(TCanvas( name, name, 200+10*f, 200+10*f, 1000, 600 ))
    cansingle[f].SetFillColor( 0 )
    cansingle[f].Divide( 1,1 )
    cansingle[f].cd(1)


histos = []
text1 = []
for f in range(0,8):
  histos.append([])
  text1.append([])
  for i in range(0,16):
    histos[f].append([])
    text1[f].append([])
    for j in range(0,3):
      histos[f][i].append([])
      text1[f][i].append([])
      for k in range(0,6):
        histos[f][i][j].append([])
        text1[f][i][j].append([])

datasettext = ["Data", "Off peak", "Signal0", "Signal1", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}"] 

# 
# Define some variable ranges
#
lox = 5.15
hix = 5.30
loy = -0.20
hiy =  0.20
if whichType == "hmass":
  if whichPlot==0 or whichPlot==2:
    lox = 5.20
    hix = 5.30
    if drange:
      lox = 1.85
      hix = 1.90
  elif whichPlot==1:
    if ( baryon=="Lambda0"):
      lox = 1.105
      hix = 1.125
    elif ( baryon=="LambdaC"):
      lox = 2.2
      hix = 2.4
  elif whichPlot==3:
    lox = -0.2
    hix = 0.2
  elif whichPlot==4 or whichPlot==6 or whichPlot==8:
    lox = 0.0
    hix = 8.0
  elif whichPlot==5 or whichPlot==7 or whichPlot==9:
    lox = -1.0
    hix = 1.0
elif whichType == "hshape":
  lox = -1.0
  hix = 1.0
  if whichPlot==3:
    lox = -0.5
    hix = 2.0
  if whichPlot==4:
    lox = -5.0
    hix = 20.0
  elif whichPlot==5:
    lox = -5.0
    hix = 20.0
  elif whichPlot==7 or whichPlot==6:
    lox = 0.0
    hix = 1.0
  elif whichPlot==10:
    lox = 4.0
    hix = 7.0
  elif whichPlot==11:
    lox = 0.0
    hix = 1.0
elif whichType == "h2d":
  if whichPlot==0:
    lox = 5.20
    hix = 5.30
    loy = -0.20
    hiy = 0.20
    if drange:
      lox = 1.85
      loy = 1.90
 

xaxistitle = "X-axis"
yaxistitle = "Y-axis"
#### Mass variables
if whichType == "hmass":
  if whichPlot==0:
    xaxistitle = "M(baryon lepton) GeV/c^{2}"
  elif whichPlot==1:
    if ( baryon=="Lambda0"):
      xaxistitle = "M(#Lambda^{0}) GeV/c^{2}"
    elif ( baryon=="LambdaC"):
      xaxistitle = "M(#Lambda_{C}^{+}) GeV/c^{2}"
  elif whichPlot==2:
    xaxistitle = "mES GeV/c^{2}"
  elif whichPlot==3:
    xaxistitle = "#Delta E GeV"
  elif whichPlot==4:
    xaxistitle = "B cand |p| GeV/c"
  elif whichPlot==5:
    xaxistitle = "B cand cos(#theta)"
  elif whichPlot==6:
    xaxistitle = "Baryon |p| GeV/c"
  elif whichPlot==7:
    xaxistitle = "Baryon cos(#theta)"
  elif whichPlot==8:
    xaxistitle = "Lepton |p| GeV/c"
  elif whichPlot==9:
    xaxistitle = "Lepton cos(#theta)"
#### Shape variables
elif whichType == "hshape":
  if whichPlot==0:
    xaxistitle = "Sphericity"
  elif whichPlot==1:
    xaxistitle = "Cos(#theta) Sphericity "
  elif whichPlot==2:
    xaxistitle = "Cos(#theta) Thrust "
  elif whichPlot==3:
    xaxistitle = "Cos(#theta) Thrust "
  elif whichPlot==4:
    xaxistitle = "Legendre polynomial 0"
  elif whichPlot==5:
    xaxistitle = "Legendre polynomial 2"
  elif whichPlot==6:
    xaxistitle = "Sphericity"
  elif whichPlot==7:
    xaxistitle = "Sphericity ROE"
  elif whichPlot==8:
    xaxistitle = "Thrust"
  elif whichPlot==9:
    xaxistitle = "Thrust ROE"
  elif whichPlot==10:
    xaxistitle = "Pre-fit Mass GeV/c^{2}"
  elif whichPlot==11:
    xaxistitle = "R2All"
elif whichType == "h2d":
  if whichPlot==0:
    xaxistitle = "mES GeV/c^{2}"
    yaxistitle = "#Delta E GeV"

#
# Open a ROOT file and save the formula, function and histogram
#
#LambdaC_SP1005_unblind_conLambdaC_ntp1.root
filename = []
for f in range(0,8):
  name = ""
  if f==0:
    name = baryon+"_data_ntp"+ntp+".root"
  elif f==1:
    #name = baryon+"_data_OffPeak_ntp"+ntp+".root"
    name = baryon+"_data_OffPeak_unblind_conLambdaC_ntp"+ntp+".root"
  else:
    name = baryon+"_SP"+mode[f-2]+"_unblind_conLambdaC_ntp"+ntp+".root"
  filename.append(name)

for f in range(0,8):
  print filename[f]
  rootfile = TFile( filename[f] )
  if os.path.isfile(filename[f] ):
    #print f
    print filename[f]
    for i in range(whichPlot, whichPlot+1):
      for j in range(0,3):
        for k in range(0,6):
          if(j<2):
            hname = whichType + str(i) + "_" + str(j) + "_" + str(k)
            #print hname
            histos[f][i][j][k] = gROOT.FindObject(hname)
            if histos[f][i][j][k]:
              newname = whichType+str(f)+"_" + str(i) + "_" + str(j) + "_" + str(k)
              histos[f][i][j][k].SetName(newname)
            else:
              histos[f][i][j][k] = TH1F(hname,hname,10,0,1)

          else:
            hname = whichType + str(i) + "_" + str(0) + "_" + str(k)
            #print hname
            histos[f][i][j][k] = histos[f][i][0][k] 
            histos[f][i][j][k].Add(histos[f][i][1][k] )
            if histos[f][i][j][k]:
              newname = whichType+str(f)+"_" + str(i) + "_" + str(j) + "_" + str(k)
              histos[f][i][j][k].SetName(newname)
            else:
              histos[f][i][j][k] = TH1F(hname,hname,10,0,1)

          # Draw the canvas labels
          
          bottompad[j].cd(8*k + f + 1)
          if whichPlot!=1:
            """
            if(j<2 and whichType=="hmass" and not drange):
              histos[f][i][j][k].Rebin(2)
            if(j<2 and whichType=="h2d"):
              histos[f][i][j][k].RebinX(5)
              histos[f][i][j][k].RebinY(5)
              """
            print "0 here " + str(hix)
            histos[f][i][j][k].GetXaxis().SetRangeUser(lox, hix)
            hmax = 2.0 * histos[f][i][j][k].GetMaximum()
            #print hmax
            if whichType=="hmass":
              histos[f][i][j][k].SetMaximum(hmax)

          histos[f][i][j][k].SetMinimum(0)
          histos[f][i][j][k].SetTitle("")
          
          histos[f][i][j][k].GetYaxis().SetNdivisions(4)
          histos[f][i][j][k].GetXaxis().SetNdivisions(6)
          histos[f][i][j][k].GetYaxis().SetLabelSize(0.06)
          histos[f][i][j][k].GetXaxis().SetLabelSize(0.06)

          histos[f][i][j][k].GetXaxis().CenterTitle()
          histos[f][i][j][k].GetXaxis().SetTitleSize(0.09)
          histos[f][i][j][k].GetXaxis().SetTitleOffset(1.0)
          histos[f][i][j][k].GetXaxis().SetTitle(xaxistitle)

          if whichPlot=="h2d":
            histos[f][i][j][k].GetrYaxis().SetTitle(yaxistitle)

          histos[f][i][j][k].SetFillColor(2 + (f)/2)
          print "here " + str(hix)
          histos[f][i][j][k].GetXaxis().SetRangeUser(lox, hix)
          
          if whichType != "h2d":
            #print str(f)+" "+str(i)+" "+str(j)+" "+str(k)
            ##print histos[f][i][j][k]
            histos[f][i][j][k].DrawCopy()
          else:
            histos[f][i][j][k].DrawCopy("colz")

          if len(index)==2:
            if j == index[0] and k == index[1]:
              cansingle[f].cd(1)
              #print f
              #print i
              #print j
              #print k
              #print histos[f][i][j][k]
              if histos[f][i][j][k]:
                if whichType != "h2d":
                  histos[f][i][j][k].DrawCopy()
                else:
                  histos[f][i][j][k].DrawCopy("colz")

              cansingle[f].Update()

          bottompad[j].cd(8*k + f + 1)
          text1[f][i][j][k] = TPaveText(0.0,0.8,0.4,1.0,"NDC")
          text1[f][i][j][k].AddText(datasettext[f])
          text1[f][i][j][k].SetBorderSize(1)
          text1[f][i][j][k].SetFillStyle(1)
          text1[f][i][j][k].Draw()

          can[j].Update()



if len(index)==2:
  for f in range(0,5):
    name = "plots/cansingle" + str(f) + "_" +str(index[0])+"_"+str(index[1])+"_" + baryon + "_" + ntp + "_" + whichType + "_" + str(whichPlot) + ".eps" 
    cansingle[f].SaveAs(name)


for j in range(0,3):
  name = "plots/can" + str(j) + "_" + baryon + "_" + ntp + "_" + whichType + "_" + str(whichPlot) + ".ps" 
  can[j].SaveAs(name)

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
