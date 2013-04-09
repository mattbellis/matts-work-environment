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
tag = sys.argv[2]
testBack = sys.argv[3]

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
pselectors = []
kselectors = []
piselectors = []

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






canvastext = []
can = []
hleg = []
toppad = []
bottompad = []
for f in range(0,1):
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
  bottompad[f].Divide(7, 7)

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(pselectors[f])
  canvastext[f].AddText("Selectors")
  canvastext[f].AddText(tag)
  #canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1001)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()


hbase = []

histos = []
text1 = []
text2 = []
for f in range(0,6):
  histos.append([])
  text1.append([])
  text2.append([])
  for i in range(0,6):
    histos[f].append([])
    text1[f].append([])
    text2[f].append([])
    for j in range(0,6):
      histos[f][i].append([])
      text1[f][i].append([])
      text2[f][i].append([])

datasettext = ["Data", "Off peak", "Signal0", "Signal1", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}"] 

# 
# Define some variable ranges
#
#lox = 2.2
#hix = 2.4


xaxistitle = "X-axis"
yaxistitle = "Y-axis"
#### Mass variables
xaxistitle = "M(#Lambda^{0}) GeV/c^{2}"
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
    for k in range(0,3):
      hname = "hmass0_0_" + str(k+3)
      hbase.append(gROOT.FindObject(hname))
      if (k == 0):
        hmax = 1.2 * hbase[k].GetMaximum()
        print hmax
      hbase[k].SetMaximum(hmax)

      hbase[k].SetFillColor(k + 1)

      hbase[k].SetMinimum(0)
      hbase[k].SetTitle("")
      
      hbase[k].GetYaxis().SetNdivisions(4)
      hbase[k].GetYaxis().CenterTitle()
      hbase[k].GetYaxis().SetLabelSize(0.06)
      hbase[k].GetYaxis().SetTitleOffset(0.9)

      hbase[k].GetXaxis().SetNdivisions(6)
      hbase[k].GetXaxis().CenterTitle()
      hbase[k].GetXaxis().SetLabelSize(0.06)
      hbase[k].GetXaxis().SetTitleSize(0.09)
      hbase[k].GetXaxis().SetTitleOffset(1.0)
      hbase[k].GetXaxis().SetTitle(xaxistitle)

    ################################
    # Get the rest
    ################################
    for i in range(0,6):
      for j in range(0,6):
        for k in range(0,1):
          hname = "hmass0_0_" + str(6*i + j + 3)
          #print hname
          histos[i][j][k] = gROOT.FindObject(hname)
          if histos[i][j][k]:
            histos[i][j][k].SetName(hname)
          else:
            histos[i][j][k] = TH1F(hname,hname,10,0,1)
            histos[i][j][k].SetName(hname)

          # Draw the canvas labels
          
          #bottompad[i].cd(6*j + k + 8 + (j+1))
          bottompad[0].cd(6*i + j + 8 + (i+1))
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
          if i==0 and j==0 and k==0:
            histos[i][j][k].SetFillColor(2)
          else:
            histos[i][j][k].SetFillColor(5)
          #print "here " + str(hix)
          #histos[i][j][k].GetXaxis().SetRangeUser(lox, hix)
          hbase[0].Draw()
          #hbase[2].Draw("same")
          histos[i][j][k].Draw("same")

          if j==1 and k==0:
            bottompad[0].cd(1)
            hleg.append(TLegend(0.01, 0.01, 0.99, 0.99))
            print "legcount: " + str(legcount)
            hleg[legcount].AddEntry(hbase[0],"m_{ES}/#Delta E range","f")
            #hleg[legcount].AddEntry(hbase[2],"P(#Chi^{2})>0.001","f")
            hleg[legcount].AddEntry(histos[0][1][0],"Selectors cut","f")
            hleg[legcount].Draw()
            legcount += 1
            gPad.Update()


          count += 1
          
      ######################
      # Draw labels
      ######################
      #"""
      for j in range(0,6):
        for k in range(0,6):
          if j==0 or k==0:
            #bottompad[i].cd(7*j + k + 1)
            #text1[i][j][k] = TPaveText(0.0,0.0,0.99,0.99,"NDC")
            if k==0:
              bottompad[0].cd(7*j + 8)
              print str(7*j+8) + " " + str(j) + str(k) + " " + kselectors[j]
              text1[i][j][k] = TPaveText(0.01,0.01,0.99,0.99,"NDC")
              text1[i][j][k].AddText(pselectors[j])
              text1[i][j][k].SetBorderSize(1)
              text1[i][j][k].SetFillStyle(1)
              text1[i][j][k].Draw()
            if j==0:
              bottompad[0].cd(k + 2)
              text2[i][j][k] = TPaveText(0.01,0.01,0.99,0.99,"NDC")
              text2[i][j][k].AddText(piselectors[k])
              text2[i][j][k].SetBorderSize(1)
              text2[i][j][k].SetFillStyle(1)
              text2[i][j][k].Draw()
            #"""

      gPad.Update()

############################################
# Maybe do a fit
############################################
#lammass = 2.284
lammass = 1.11589
outfile = open("outVals_"+tag+".txt","w+")
#lamrangelo = lammass - 0.004
#lamrangehi = lammass + 0.004
lamrangelo = lammass - 0.0025
lamrangehi = lammass + 0.0025
#lox = 1.105
#hix = 1.125
lox = 1.110
hix = 1.120
lobin = 1
hibin = 1
lodiffmin = 10000.0
hidiffmin = 10000.0
for i in range(1,histos[0][0][0].GetNbinsX() + 1):
  lodiff = abs(lamrangelo - histos[0][0][0].GetBinCenter(i) )
  if lodiff < lodiffmin:
    lodiffmin = lodiff
    lobin = i+1
  hidiff = abs(lamrangehi - histos[0][0][0].GetBinCenter(i) )
  if hidiff < hidiffmin:
    hidiffmin = hidiff
    hibin = i

binwidth = histos[0][0][0].GetBinWidth(1)

print "lobin: " + str(lobin)
print "hibin: " + str(hibin)

fit = []
gfit = []
pfit = []

bfit = []
bgfit = []
bpfit = []

testfit = TF1("testfit", "gaus(0)+pol1(3)", lox, hix)

totevents = []
sigevents = []
bakevents = []

btotevents = []
bsigevents = []
bbakevents = []


#################################
# Do the base first
#################################
count = 0
for i in range(0,3):
  bottompad[0].cd(9)
  bfit.append(TF1( "bfit"+str(count), "gaus(0)+pol1(3)", lox, hix))
  bgfit.append(TF1("bgfit"+str(count), "gaus", lamrangelo, lamrangehi))
  bpfit.append(TF1("bpfit"+str(count), "pol1", lamrangelo, lamrangehi))
  ############################################
  # Maybe do a fit
  ############################################
  testfit.SetParameter(1, lammass)
  testfit.SetParameter(2, 0.005)
  #bfit[count].SetParameter(1, lammass)
  #bfit[count].SetParameter(2, 0.005)
  # Fit
  hbase[i].Fit("testfit", "ERFQN","", 1.105, 1.125)
  # Redraw
  hbase[i].Draw()
  # Draw the fit results
  bfit[count].SetParameter(0, testfit.GetParameter(0))
  bfit[count].SetParameter(1, testfit.GetParameter(1))
  bfit[count].SetParameter(2, testfit.GetParameter(2))
  bfit[count].SetParameter(3, testfit.GetParameter(3))
  bfit[count].SetParameter(4, testfit.GetParameter(4))
  bfit[count].SetLineColor(2)
  bfit[count].SetLineWidth(2)
  bfit[count].Draw("same")
  bgfit[count].SetParameter(0, testfit.GetParameter(0))
  bgfit[count].SetParameter(1, testfit.GetParameter(1))
  bgfit[count].SetParameter(2, testfit.GetParameter(2))
  bgfit[count].SetLineColor(4)
  bgfit[count].SetLineWidth(2)
  #bgfit[count].Draw("same")
  bpfit[count].SetParameter(0, testfit.GetParameter(3))
  bpfit[count].SetParameter(1, testfit.GetParameter(4))
  bpfit[count].SetLineColor(3)
  bpfit[count].SetLineWidth(2)
  bpfit[count].Draw("same")
  numback = bpfit[count].Integral(lamrangelo, lamrangehi)/binwidth
  numtot = hbase[i].Integral(lobin, hibin)
  bbakevents.append(numback)
  btotevents.append(numtot)
  bsigevents.append(numtot-numback)
  print "Base ============== "
  print "back: " + str(numback)
  print "tot:  " + str(numtot)
  print "sig:  " + str(numtot - numback)
  outfile.write(hbase[i].GetName() + " " + str(numtot) + " " + str(numback) + "\n")
  count += 1

  gPad.Update()

whichfittouse = 0
maxsig = bsigevents[whichfittouse]
maxbak = bbakevents[whichfittouse]

count = 0
for i in range(0,6):
  totevents.append([])
  sigevents.append([])
  bakevents.append([])
  for j in range(0,6):
    totevents[i].append([])
    sigevents[i].append([])
    bakevents[i].append([])
    for k in range(0,1):
      #bottompad[0].cd(6*i + j + 8 + (i+1))
      bottompad[0].cd(6*i + j + 8 + (i+1))
      fit.append(TF1( "fit"+str(count), "gaus(0)+pol1(3)", lox, hix))
      gfit.append(TF1("gfit"+str(count), "gaus", lamrangelo, lamrangehi))
      pfit.append(TF1("pfit"+str(count), "pol1", lamrangelo, lamrangehi))
      ############################################
      # Maybe do a fit
      ############################################
      testfit.SetParameter(1, lammass)
      testfit.SetParameter(2, 0.005)
      #fit[count].SetParameter(1, lammass)
      #fit[count].SetParameter(2, 0.005)
      # Fit
      histos[i][j][k].Fit("testfit", "ERFQN","", 1.105, 1.125)
      # Redraw
      #print hbase[0].GetFunction("fit")
      hbase[0].Draw()
      #hbase[2].Draw("same")
      #hbase[2].Draw()
      bfit[whichfittouse].Draw("same")
      bgfit[whichfittouse].Draw("same")
      bpfit[whichfittouse].Draw("same")
      histos[i][j][k].Draw("same")
      # Draw the fit results
      fit[count].SetParameter(0, testfit.GetParameter(0))
      fit[count].SetParameter(1, testfit.GetParameter(1))
      fit[count].SetParameter(2, testfit.GetParameter(2))
      fit[count].SetParameter(3, testfit.GetParameter(3))
      fit[count].SetParameter(4, testfit.GetParameter(4))
      fit[count].SetLineColor(2)
      fit[count].SetLineWidth(2)
      fit[count].Draw("same")
      gfit[count].SetParameter(0, testfit.GetParameter(0))
      gfit[count].SetParameter(1, testfit.GetParameter(1))
      gfit[count].SetParameter(2, testfit.GetParameter(2))
      gfit[count].SetLineColor(4)
      gfit[count].SetLineWidth(2)
      gfit[count].Draw("same")
      pfit[count].SetParameter(0, testfit.GetParameter(3))
      pfit[count].SetParameter(1, testfit.GetParameter(4))
      pfit[count].SetLineColor(3)
      pfit[count].SetLineWidth(2)
      pfit[count].Draw("same")
      numback = pfit[count].Integral(lamrangelo, lamrangehi)/binwidth
      numtot = histos[i][j][k].Integral(lobin, hibin)
      bakevents[i][j].append(numback)
      totevents[i][j].append(numtot)
      if onlyBack:
        sigevents[i][j].append(maxsig)
      else:
        sigevents[i][j].append(numtot-numback)
      print str(lox) + " " + str(hix)
      print "---------------"
      print "back: " + str(maxbak)
      print "sig:  " + str(maxsig)
      print "---"
      print "tot:  " + str(numtot)
      print "back: " + str(numback)
      print "sig:  " + str(numtot - numback)
      outfile.write(histos[i][j][k].GetName() + " " + str(numtot) + " " + str(numback) + "\n")
      count += 1

  # Save histograms plots
  savename = "Plots/lambda0_mass_plots" + tag + "_" + str(f) + ".eps"
  can[0].SaveAs(savename)
  gPad.Update()


##################################
# Make some plots for a talk
##################################
i=3
j=3
k=0
fitcount = 36*i + 6*j + k
cantalk = []

# Cutaway region
htemp = TH1F()
hbase[0].Copy(htemp)
for p in range(1, htemp.GetNbinsX()+1):
  if p<lobin or p>hibin:
    htemp.SetBinContent(p, 0)
htemp.SetFillColor(45)

# Cut away region
htemp2 = TH1F()
histos[i][j][k].Copy(htemp2)
for p in range(1, htemp2.GetNbinsX()+1):
  if p<lobin or p>hibin:
    htemp2.SetBinContent(p, 0)
htemp2.SetFillColor(45)

# Make the plots
for f in range(0,10):
  cantalk.append(TCanvas("cantalk"+str(f),"",250+10*f,250+10*f,600,600))
  cantalk[f].SetFillColor(0)
  cantalk[f].Divide(1,1)
  cantalk[f].cd(1)
  i=3
  j=3
  k=0
  #fitcount = 36*i + 6*j + k
  fitcount = 6*i + j + k
  if f==0:
    hbase[0].Draw()
  elif f==1:
    hbase[0].DrawCopy()
    bfit[whichfittouse].Draw("same")
    #bgfit[whichfittouse].Draw("same")
    #bpfit[whichfittouse].Draw("same")
  elif f==2:
    hbase[0].DrawCopy()
    htemp.Draw("same")

    bfit[whichfittouse].Draw("same")
    #bgfit[whichfittouse].Draw("same")
    #bpfit[whichfittouse].Draw("same")
  elif f==3:
    hbase[0].DrawCopy()
    htemp.Draw("same")

    # Draw the box
    box = TBox(lamrangelo, 0.0, lamrangehi, bpfit[whichfittouse].Eval(lammass))
    box.SetFillStyle(3008)
    box.SetFillColor(3)
    box.Draw()
    bfit[whichfittouse].Draw("same")
    #bgfit[whichfittouse].Draw("same")
    bpfit[whichfittouse].Draw("same")

  elif f==4:
    histos[i][j][k].Draw("")
  elif f==5:
    histos[i][j][k].Draw("")
    fit[fitcount].Draw("same")
    #gfit[fitcount].Draw("same")
    #pfit[fitcount].Draw("same")

  elif f==6:
    histos[i][j][k].Draw("")
    htemp2.Draw("same")

    fit[fitcount].Draw("same")
    #gfit[fitcount].Draw("same")
    #pfit[fitcount].Draw("same")

  elif f==7:
    histos[i][j][k].Draw("")
    htemp2.Draw("same")

    # Draw the box
    box2 = TBox(lamrangelo, 0.0, lamrangehi, pfit[fitcount].Eval(lammass))
    box2.SetFillStyle(3008)
    box2.SetFillColor(3)
    box2.Draw()
    fit[fitcount].Draw("same")
    #gfit[fitcount].Draw("same")
    pfit[fitcount].Draw("same")
  elif f==8:
    hbase[0].DrawCopy()
    bfit[whichfittouse].Draw("same")
    bpfit[whichfittouse].Draw("same")
    histos[i][j][k].Draw("same")
    fit[fitcount].Draw("same")
    pfit[fitcount].Draw("same")
  elif f==9:
    i=4
    j=2
    k=0
    #fitcount = 36*i + 6*j + k
    fitcount = 6*i + j + k
    hbase[0].SetMaximum(550.0)
    hbase[0].DrawCopy()
    bfit[whichfittouse].Draw("same")
    bpfit[whichfittouse].Draw("same")
    histos[i][j][k].Draw("same")
    fit[fitcount].Draw("same")
    pfit[fitcount].Draw("same")
  else:
    histos[i][j][k].Draw("")

  cantalk[f].Update()
  savename = "Plots/cantalk_individual_fit_histos_" + tag + "_" + str(f) + ".eps"
  cantalk[f].SaveAs(savename)


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
