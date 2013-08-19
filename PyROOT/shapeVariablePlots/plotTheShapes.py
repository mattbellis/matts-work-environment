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
infilename = sys.argv[1]
reffilename = sys.argv[2]
tag = sys.argv[3]

numshapes = 32
numcuts = 7

basehisto = 6

histos_to_divide = [0, 1, 2, 3, 4, 5]
numcuts = len(histos_to_divide) 

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
gStyle.SetPadBottomMargin(0.23)
gStyle.SetPadColor(0)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)


canvastitles = ["B", "B","B", "B", "B", "B"]


####################################################
# Selectors!
####################################################
pselectors = []
kselectors = []
piselectors = []
muselectors = []

pselectors.append("SuperLooseKMProtonSelection")
pselectors.append("VeryLooseKMProtonSelection")
pselectors.append("LooseKMProtonSelection")
pselectors.append("TightKMProtonSelection")
pselectors.append("VeryTightKMProtonSelection")
pselectors.append("SuperTightKMProtonSelection")

kselectors.append("SuperLooseKMKaonMicroSelection")
kselectors.append("VeryLooseKMKaonMicroSelection")
kselectors.append("LooseKMKaonMicroSelection")
kselectors.append("TightKMKaonMicroSelection")
kselectors.append("VeryTightKMKaonMicroSelection")
kselectors.append("SuperTightKMKaonMicroSelection")

piselectors.append("SuperLooseKMPionSelection")
piselectors.append("VeryLooseKMPionSelection")
piselectors.append("LooseKMPionSelection")
piselectors.append("TightKMPionSelection")
piselectors.append("VeryTightKMPionSelection")
piselectors.append("SuperTightKMPionSelection")

muselectors.append("BDTVeryLooseMuonSelection")
muselectors.append("BDTLooseMuonSelection")
muselectors.append("BDTTightMuonSelection")
muselectors.append("BDTVeryTightMuonSelection")
muselectors.append("BDTVeryLooseMuonSelectionFakeRate")
muselectors.append("BDTLooseMuonSelectionFakeRate")
muselectors.append("BDTTightMuonSelectionFakeRate")
muselectors.append("BDTVeryTightMuonSelectionFakeRate")

#####################################################

canvastext = []
can = []
hleg = []
toppad = []
bottompad = []
for f in range(0,numcuts + 2):
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
  bottompad[f].Divide(4, 7)

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
href = []

legpct = []
histos = []
text1 = []
text2 = []
for f in range(0,1):
  histos.append([])
  legpct.append([])
  text1.append([])
  text2.append([])
  for i in range(0,numshapes):
    hbase.append([])
    href.append([])
    legpct[f].append([])
    histos[f].append([])
    text1[f].append([])
    text2[f].append([])
    for j in range(0,numcuts):
      histos[f][i].append([])
      legpct[f][i].append([])
      text1[f][i].append([])
      text2[f][i].append([])

datasettext = ["Data", "Off peak", "Signal0", "Signal1", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}"] 
#outfile = open("outVals_"+tag+".txt","w+")

# 
# Define some variable ranges
#
#lox = 2.2
#hix = 2.4

whichpad = [-1, -1, 1,    5,6,7,8,    9,10,11,12,  2,3,4, 13,14,15,16, 17,18,19,20, 21, -1, 25, -1, 26, -1,-1, -1, 27, 28]  


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
  print reffilename
  refrootfile = TFile( reffilename )
  refrootfile = TFile( reffilename )
  if os.path.isfile(reffilename ):
    #print f
    ################################
    # Get the first three histos as they are different
    ################################
    for j in range(0,1):
      for k in range(0,numcuts+1):
        if k == 0:
          hname = "h2D0_" + str(1) + "_" + str( basehisto )
        else:
          hname = "h2D0_" + str(1) + "_" + str( histos_to_divide[k-1] )
        href[j].append(gROOT.FindObject(hname))
        print hname
        #if (k == 0):
          #hmax = 1.65 * href[j][k].GetMaximum()
          #print hmax
        #href[j][k].SetMaximum(hmax)

        href[j][k].SetFillColor(44)

        href[j][k].SetMinimum(0)
        href[j][k].SetTitle("")
        
        href[j][k].GetYaxis().SetNdivisions(4)
        href[j][k].GetYaxis().CenterTitle()
        href[j][k].GetYaxis().SetLabelSize(0.06)
        href[j][k].GetYaxis().SetTitleOffset(0.9)

        href[j][k].GetXaxis().SetNdivisions(6)
        href[j][k].GetXaxis().CenterTitle()
        href[j][k].GetXaxis().SetLabelSize(0.06)
        href[j][k].GetXaxis().SetTitleSize(0.10)
        href[j][k].GetXaxis().SetTitleOffset(0.9)

        href[j][k].GetListOfFunctions().FindObject("palette").SetX1NDC(0.90)
        href[j][k].GetListOfFunctions().FindObject("palette").SetX2NDC(0.95)
        href[j][k].GetListOfFunctions().FindObject("palette").SetY1NDC(0.20)
        href[j][k].GetListOfFunctions().FindObject("palette").SetY2NDC(0.90)


  print infilename
  rootfile = TFile( infilename )
  rootfileref = TFile( infilename )
  if os.path.isfile(infilename ):
    #print f
    ################################
    # Get the first three histos as they are different
    ################################
    for j in range(0,numshapes):
      for k in range(0,1):
        hname = "hshape0_" + str(j) + "_" + str( basehisto )
        hbase[j].append(gROOT.FindObject(hname))
        print hname
        if (k == 0):
          hmax = 1.65 * hbase[j][k].GetMaximum()
          #print hmax
        #hbase[j][k].SetMaximum(hmax)

        hbase[j][k].SetFillColor(44)

        hbase[j][k].SetMinimum(0)
        hbase[j][k].SetTitle("")
        
        hbase[j][k].GetYaxis().SetNdivisions(4)
        hbase[j][k].GetYaxis().CenterTitle()
        hbase[j][k].GetYaxis().SetLabelSize(0.06)
        hbase[j][k].GetYaxis().SetTitleOffset(0.9)

        hbase[j][k].GetXaxis().SetNdivisions(6)
        hbase[j][k].GetXaxis().CenterTitle()
        hbase[j][k].GetXaxis().SetLabelSize(0.06)
        hbase[j][k].GetXaxis().SetTitleSize(0.16)
        hbase[j][k].GetXaxis().SetTitleOffset(0.6)

    ################################
    # Get the rest
    ################################
    for i in range(0,1):
      for j in range(0,numshapes):
        for k in range(0,numcuts):
          hname = "hshape0_" + str(j) + "_" + str( histos_to_divide[k] )
          #print hname
          histos[i][j][k] = gROOT.FindObject(hname)
          if histos[i][j][k]:
            histos[i][j][k].SetName(hname)
          else:
            histos[i][j][k] = TH1F(hname,hname,10,0,1)
            histos[i][j][k].SetName(hname)

          # Draw the canvas labels
          
          if whichpad[j] >= 0:
            bottompad[0].cd(whichpad[j])
          #histos[i][j][k].SetMaximum(hmax)

          histos[i][j][k].SetMinimum(0)
          histos[i][j][k].SetTitle("")
          
          histos[i][j][k].GetYaxis().CenterTitle()
          histos[i][j][k].GetYaxis().SetNdivisions(4)
          histos[i][j][k].GetYaxis().SetLabelSize(0.06)
          histos[i][j][k].GetYaxis().SetTitleOffset(0.9)

          histos[i][j][k].GetXaxis().SetNdivisions(6)
          histos[i][j][k].GetXaxis().CenterTitle()
          histos[i][j][k].GetXaxis().SetLabelSize(0.06)
          histos[i][j][k].GetXaxis().SetTitleSize(0.16)
          histos[i][j][k].GetXaxis().SetTitleOffset(0.6)
          #histos[i][j][k].GetXaxis().SetTitle(xaxistitle)

          #histos[i][j][k].SetFillColor(2 + 36*i + 6*i + k)
          color = 2+k
          if color == 5:
            color = 8
          histos[i][j][k].SetFillColor(color)
          histos[i][j][k].SetMarkerColor(color)
          histos[i][j][k].SetLineColor(color)

          histos[i][j][k].SetLineWidth(1)
          #print "here " + str(hix)
          #histos[i][j][k].GetXaxis().SetRangeUser(lox, hix)
          if whichpad[j] >= 0:
            bottompad[0].cd(whichpad[j])
            if j>=2:
              if k==0:
                hbase[j][0].Draw()
              #histos[i][j][k].DrawCopy("same")
              can[0].Update()

              ###############
              # Print individually
              ###############
              bottompad[k+1].cd(whichpad[j])
              histos[i][j][k].DrawCopy()
              bottompad[k+1].Update()


      """
          if k==0:
            bottompad[0].cd(1)
            hleg.append(TLegend(0.01, 0.01, 0.99, 0.99))
            print "legcount: " + str(legcount)
            hleg[legcount].AddEntry(hbase[j][0],"m_{ES}/#Delta E range","f")
            hleg[legcount].AddEntry(histos[0][0][0],"Selectors cut","f")
            hleg[legcount].Draw()
            legcount += 1


          count += 1
      """
          
      ######################
      # Draw labels
      ######################
      """
      for j in range(0,3):
        for k in range(0,numcuts):
          bottompad[0].cd(j+1)
          print str(7*j+8) + " " + str(j) + str(k) + " " + muselectors[k]
          text1[i][j][k] = TPaveText(0.01,0.01,0.99,0.60,"NDC")
          text1[i][j][k].AddText(muselectors[k])
          text1[i][j][k].SetBorderSize(1)
          text1[i][j][k].SetFillStyle(1)
          text1[i][j][k].Draw()
      """

      ######################
      # Draw legends
      ######################
      """
      for j in range(0,3):
        for k in range(0,numcuts):
          if k<8:
            bottompad[0].cd(j+1)
            legpct[i][j][k] = TLegend(0.01, 0.75, 0.60, 0.99)
            num0 = float(hbase[j][0].GetEntries())
            num  = float(histos[i][j][k].GetEntries())
            print str(i) + " " + str(j) + " " + str(k) + " " + str(num0) + " " + str(num)
            if num0!=0:
              words =  "Entries: %d" % (num0)
              legpct[i][j][k].AddEntry(hbase[j][0],     words, "f")
              words =  "%s %2.1f" % ("%", 100*num/num0)
              legpct[i][j][k].AddEntry(histos[i][j][k], words, "f")
              #print words
              #outfile.write(histos[i][j][k].GetName() + " " + str(num0) + " " + str(num) + "\n")
            legpct[i][j][k].Draw()

        can[0].Update()
      """

hdiv = []
for i in range(0,1):
  hdiv.append([])
  for j in range(0,numshapes):
    hdiv[i].append([])
    for k in range(0,numcuts):
      #print j
      if j>=2:
        if whichpad[j] >= 0:
          bottompad[numcuts + 1].cd(whichpad[j])
          #hdum =  histos[i][j][0].Clone()
          #hdum2 = histos[i][j][1].Clone()
          hdiv[i][j].append(histos[i][j][k].Clone())
          #hdiv[i][j][k].Rebin(2)

          hdiv[i][j][k].Sumw2()

          ######################################
          # Divide and draw
          ######################################
          scale = float(hbase[j][0].GetEntries())/histos[i][j][k].GetEntries()
          print str(k) + " scale: " + str(scale)
          hdiv[i][j][k].Scale(scale)
          #hdiv[i][j][k].Sumw2()
          hdiv[i][j][k].Divide(hbase[j][0])

          hdiv[i][j][k].SetMinimum(0.0)
          hdiv[i][j][k].SetMaximum(2.0)
          #if k==0:
          if k==0:
            hdiv[i][j][k].DrawCopy("e")
          elif k>0:
            hdiv[i][j][k].DrawCopy("samee")
          gPad.Update()

###########################
# Make some individual plots
###########################
cantalk = []
sigbox = []
line0 = []
plotcount = 0
for j in range(0,numshapes):
  cantalk.append([])
  sigbox.append([])
  line0.append([])
  if whichpad[j] >= 0:
    for k in range(0,numcuts):
      if k==0:
        name = "cantalk" + str(j) + "_" + str(k)
        #cantalk[j].append(TCanvas( name, name, 50*k+10*j, 10+10*j, 800, 300 ))
        #cantalk[j].append(TCanvas( name, name, 50*k+10*j, 10+10*j, 1200, numcuts*300 ))
        cantalk[j].append(TCanvas( name, name, 50*k+10*j, 10+10*j, 1400, 700 ))
        cantalk[j][k].SetFillStyle(1001)
        cantalk[j][k].SetFillColor(0)
        #cantalk[j][k].Divide(2,1)
        cantalk[j][k].Divide(3,numcuts)

      #cantalk[j][0].cd(1 + 4*k)
      #href[0][0].DrawCopy("colz")

      cantalk[j][0].cd(1 + 3*k)
      href[0][k+1].DrawCopy("colz")
      sigbox[j].append(TBox( 5.2, -0.3, 5.3, 0.2 ) )
      sigbox[j][k].SetLineColor(kYellow)
      sigbox[j][k].SetLineWidth(4)
      sigbox[j][k].SetFillColor(0)
      sigbox[j][k].SetFillStyle(0)
      sigbox[j][k].Draw()

      cantalk[j][0].cd(2 + 3*k)
      hbase[j][0].DrawCopy()
      scale = float(hbase[j][0].Integral())/histos[i][j][k].Integral()
      histos[0][j][k].Scale(scale)
      histos[0][j][k].SetLineWidth(4)
      histos[0][j][k].SetFillStyle(3004)
      histos[0][j][k].DrawCopy("same")

      cantalk[j][0].cd(3 + 3*k)
      hdiv[0][j][k].DrawCopy()
      lox = hdiv[0][j][k].GetBinLowEdge(1)
      hix = hdiv[0][j][k].GetBinLowEdge( hdiv[0][j][k].GetNbinsX()+1 )
      line0[j].append(TLine(lox, 1, hix, 1))
      line0[j][k].Draw()

      cantalk[j][0].Update()

      # Print the canvases
      #name = "Plots/cantalk_shapes_" + tag + "_" + str(k) + "_" + str(plotcount) + ".eps"
      #cantalk[j][k].SaveAs(name)
      name = "Plots/cantalk_shapes_" + tag + "_" + str(0) + "_" + str(plotcount) + ".eps"
      cantalk[j][0].SaveAs(name)

    plotcount += 1


###########################
# Make the plots
###########################
for j in range(0,1):
  name = "Plots/lambdac_shapes_allplots" + tag + "_" + str(j) + "_" + ".eps" 
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
