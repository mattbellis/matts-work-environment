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
lepton = sys.argv[1]
infilename = sys.argv[2]
tag = sys.argv[3]
testBack = sys.argv[4]


plotMax = 6
if lepton == "mu" >= 0:
  plotMax = 8

onlyBack = False
if testBack == "onlyBack":
  onlyBack = True

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
#gStyle.SetOptStat(0)
gStyle.SetOptStat(10)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetPadColor(0)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)


canvastitles = ["B", "B","B", "B", "B", "B"]


####################################################
# Selectors!
####################################################
lep_selectors = []

if plotMax == 8:
  lep_selectors.append("BDTVeryLooseMuonSelection")
  lep_selectors.append("BDTLooseMuonSelection")
  lep_selectors.append("BDTTightMuonSelection")
  lep_selectors.append("BDTVeryTightMuonSelection")
  lep_selectors.append("BDTVeryLooseMuonSelectionFakeRate")
  lep_selectors.append("BDTLooseMuonSelectionFakeRate")
  lep_selectors.append("BDTTightMuonSelectionFakeRate")
  lep_selectors.append("BDTVeryTightMuonSelectionFakeRate")

else:
  lep_selectors.append("SuperLooseKMElectronMicroSelection")
  lep_selectors.append("VeryLooseKMElectronMicroSelection")
  lep_selectors.append("LooseKMElectronMicroSelection")
  lep_selectors.append("TightKMElectronMicroSelection")
  lep_selectors.append("VeryTightKMElectronMicroSelection")
  lep_selectors.append("SuperTightKMElectronMicroSelection")









canvastext = []
can = []
hleg = []
toppad = []
bottompad = []
for f in range(0,3):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1400, 900 ))
  can[f].SetFillStyle(1001)
  can[f].SetFillColor(0)
  #can[f].Divide( 5,6 )
  name = "top" + str(f)
  toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  toppad[f].SetFillStyle(1001)
  toppad[f].SetFillColor(0)
  toppad[f].Draw()
  toppad[f].Divide(1,1)
  name = "bottom" + str(f)
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  bottompad[f].SetFillColor(1001);
  bottompad[f].SetFillColor(0);
  bottompad[f].Draw();
  bottompad[f].Divide(1 + plotMax/2, 4, 0, 0);

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(pselectors[f])
  canvastext[f].AddText(tag)
  #canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1001)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()


hbase = []

legpct = []
histos = []
text1 = []
text2 = []
for f in range(0,1):
  histos.append([])
  legpct.append([])
  text1.append([])
  text2.append([])
  for i in range(0,3):
    hbase.append([])
    legpct[f].append([])
    histos[f].append([])
    text1[f].append([])
    text2[f].append([])
    for j in range(0,8):
      histos[f][i].append([])
      legpct[f][i].append([])
      text1[f][i].append([])
      text2[f][i].append([])

datasettext = ["Data", "Off peak", "Signal0", "Signal1", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}"] 
outfile = open("outVals_"+tag+".txt","w+")

# 
# Define some variable ranges
#
#lox = 2.2
#hix = 2.4


xaxistitle = "X-axis"
yaxistitle = "Y-axis"
#### Mass variables
xaxistitle = "M(#Lambda_{C}^{+}) GeV/c^{2}"
# Open a ROOT file and save the formula, function and histogram
#
#LambdaC_SP1005_unblind_conLambdaC_ntp1.root
legcount = 0
count = 0
hmax = 1.0
for f in range(0,1):
  print infilename
  rootfile = TFile( infilename )
  if os.path.isfile(infilename ):
    #print f
    ################################
    # Get the first three histos as they are different
    ################################
    for j in range(0,3):
      for k in range(0,3):
        # First one
        hname = "hmass0_" + str(j) + "_" + str(k+3)
        # 5th one
        #hname = "hmass0_" + str(j) + "_" + str(k+7)
        hbase[j].append(gROOT.FindObject(hname))
        if (k == 0):
          hmax = 1.65 * hbase[j][k].GetMaximum()
          print hmax
        hbase[j][k].SetMaximum(hmax)

        hbase[j][k].SetFillColor(k + 1)

        hbase[j][k].SetMinimum(0)
        hbase[j][k].SetTitle("")
        
        hbase[j][k].GetYaxis().SetNdivisions(4)
        hbase[j][k].GetYaxis().CenterTitle()
        hbase[j][k].GetYaxis().SetLabelSize(0.06)
        hbase[j][k].GetYaxis().SetTitleOffset(0.9)

        hbase[j][k].GetXaxis().SetNdivisions(6)
        hbase[j][k].GetXaxis().CenterTitle()
        hbase[j][k].GetXaxis().SetLabelSize(0.06)
        hbase[j][k].GetXaxis().SetTitleSize(0.09)
        hbase[j][k].GetXaxis().SetTitleOffset(1.0)
        if j==0:
          xaxistitle = "M(#Lambda_{C}^{+}) GeV/c^{2}"
        elif j==1:
          xaxistitle = "m_{ES} GeV/c^{2}"
        elif j==2:
          xaxistitle = "#Delta E GeV"
        hbase[j][k].GetXaxis().SetTitle(xaxistitle)

    ################################
    # Get the rest
    ################################
    for i in range(0,1):
      for j in range(0,3):
        for k in range(0,8):
          hname = "hmass0_" + str(j) + "_" + str(k + 3)
          print hname
          histos[i][j][k] = gROOT.FindObject(hname)
          if histos[i][j][k]:
            histos[i][j][k].SetName(hname)
          else:
            histos[i][j][k] = TH1F(hname,hname,10,0,1)
            histos[i][j][k].SetName(hname)

          # Draw the canvas labels
          
          bottompad[j].cd(k + (1 + plotMax/2)*((k/(plotMax/2)) + 1) + (k/(plotMax/2)) + 2)
          histos[i][j][k].SetMaximum(hmax)

          histos[i][j][k].SetMinimum(0)
          histos[i][j][k].SetTitle("")
          
          histos[i][j][k].GetYaxis().CenterTitle()
          histos[i][j][k].GetYaxis().SetNdivisions(4)
          histos[i][j][k].GetYaxis().SetLabelSize(0.06)
          histos[i][j][k].GetYaxis().SetTitleOffset(0.9)

          histos[i][j][k].GetXaxis().SetNdivisions(6)
          histos[i][j][k].GetXaxis().CenterTitle()
          histos[i][j][k].GetXaxis().SetTitleSize(0.09)
          histos[i][j][k].GetXaxis().SetTitleOffset(1.0)
          histos[i][j][k].GetXaxis().SetTitle(xaxistitle)

          #histos[i][j][k].SetFillColor(2 + 36*i + 6*i + k)
          if k==4:
            histos[i][j][k].SetFillColor(2)
          else:
            histos[i][j][k].SetFillColor(22)
          #print "here " + str(hix)
          #histos[i][j][k].GetXaxis().SetRangeUser(lox, hix)
          hbase[j][0].Draw()
          histos[i][j][k].Draw("same")

          can[j].Update()

          if k==0:
            bottompad[j].cd(1)
            hleg.append(TLegend(0.01, 0.01, 0.99, 0.99))
            print "legcount: " + str(legcount)
            hleg[legcount].AddEntry(hbase[j][0],"m_{ES}/#Delta E range","f")
            hleg[legcount].AddEntry(histos[0][0][0],"Selectors cut","f")
            hleg[legcount].Draw()
            legcount += 1


          count += 1
          
      ######################
      # Draw labels
      ######################
      #"""
      for j in range(0,3):
        for k in range(0, plotMax):
          bottompad[j].cd(k + (1 + plotMax/2)*((k/(plotMax/2))) + (k/(plotMax/2)) + 2)
          print str(7*j+8) + " " + str(j) + str(k) + " " + lep_selectors[k]
          text1[i][j][k] = TPaveText(0.01,0.01,0.99,0.60,"NDC")
          text1[i][j][k].AddText(lep_selectors[k])
          text1[i][j][k].SetBorderSize(1)
          text1[i][j][k].SetFillStyle(1)
          text1[i][j][k].Draw()
      #"""

      ######################
      # Draw legends
      ######################
      for j in range(0,3):
        for k in range(0,plotMax):
          if k<plotMax:
            bottompad[j].cd(k + (1+plotMax/2)*((k/(plotMax/2)) + 1) + (k/(plotMax/2)) + 2)
            legpct[i][j][k] = TLegend(0.01, 0.75, 0.60, 0.99)
            num0 = float(hbase[j][0].Integral())
            num  = float(histos[i][j][k].Integral())
            print str(i) + " " + str(j) + " " + str(k) + " " + str(num0) + " " + str(num)
            if num0!=0:
              words =  "Entries: %d" % (num0)
              legpct[i][j][k].AddEntry(hbase[j][0],     words, "f")
              words =  "%s %2.1f" % ("%", 100*num/num0)
              legpct[i][j][k].AddEntry(histos[i][j][k], words, "f")
              #print words
              outfile.write(histos[i][j][k].GetName() + " " + str(num0) + " " + str(num) + "\n")
            legpct[i][j][k].Draw()

        can[j].Update()

############################
# Save plots
###########################
for j in range(0,3):
  name = "plots/lambdac_mu_sigbkg_selectors_allplots_" + tag + "_" + str(j) + "_" + ".eps" 
  can[j].SaveAs(name)

################################################################################
## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
################################################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
