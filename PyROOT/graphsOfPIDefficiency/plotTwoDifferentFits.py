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
infiles = []
infiles.append(sys.argv[1])
infiles.append(sys.argv[2])
numsig = float(sys.argv[3])
numbak = float(sys.argv[4])
whichsig = sys.argv[5]
#lepsel = int(sys.argv[6])
tag = sys.argv[7]

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
gStyle.SetOptStat(0)
#gStyle.SetOptStat(10)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
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

lines = []
lines.append([])
lines.append([])

for f in range(0,2):
  #print infiles[f]
  file = open(infiles[f])
  for line in file:
    lines[f].append(line)

totevents = []
sigevents = []
bakevents = []

btotevents = []
bsigevents = []
bbakevents = []

for f in range(0,2):
  totevents.append([])
  sigevents.append([])
  bakevents.append([])

  # Best
  btotevents.append([])
  bsigevents.append([])
  bbakevents.append([])


##########################################
#
#
##########################################
legcount = 0
count = 0
hmax = 1.0
for f in range(0,2):
  linecount = 0
  #print infiles[f]
  if 1:
    #print f
    ################################
    # Get the first three histos as they are different
    ################################
    for k in range(0,3):
      tot = float(lines[f][linecount].split()[1])
      bak = float(lines[f][linecount].split()[2])
      btotevents[f].append(tot)
      bbakevents[f].append(bak)
      bsigevents[f].append(tot-bak)
      linecount += 1
    ################################
    # Get the rest
    ################################
    for i in range(0,6):
      totevents[f].append([])
      sigevents[f].append([])
      bakevents[f].append([])
      for j in range(0,6):
        totevents[f][i].append([])
        sigevents[f][i].append([])
        bakevents[f][i].append([])
        for k in range(0,6):
          tot = float(lines[f][linecount].split()[1])
          bak = float(lines[f][linecount].split()[2])
          totevents[f][i][j].append(tot)
          bakevents[f][i][j].append(bak)
          sigevents[f][i][j].append(tot-bak)
          linecount += 1


whichfittouse = 0
maxtot = []
maxsig = []
maxbak = []
for f in range(0,2):
  maxtot.append(btotevents[f][whichfittouse])
  maxsig.append(bsigevents[f][whichfittouse])
  maxbak.append(bbakevents[f][whichfittouse])

##############################################
# Use this for the s'/b'
##############################################
use_sprime = False
if numsig==0 and numbak==0:
  use_sprime = True
if numsig==0:
  numsig = bsigevents[0][whichfittouse]
if numbak==0:
  numbak = bbakevents[0][whichfittouse]


xpts = array('f')
ypts = array('f')

hsig = []
hbak = []
hsignif = []
grroc = [] 
tempxpts = []
tempypts = []
for g in range(0,18):
  tempxpts.append(array('f'))
  tempypts.append(array('f'))
  hsig.append(TH1D("hsig"+str(g), "", 36,0.5, 36.5))
  hbak.append(TH1D("hbak"+str(g), "", 36,0.5, 36.5))
  hsignif.append(TH1D("hsignif"+str(g), "", 36,0.5,  36.5))

##################################
# Fill with some points
##################################
for g in range(0,18):
  for i in range(0,6):
    for j in range(0,6):
      for k in range(0,6):
        xname = str(i) + str(j) + str(k)

        sigpct = sigevents[0][i][j][k]/maxsig[0]
        #bakpct = 1.0 - (bakevents[1][i][j][k]/maxbak[1])
        # Use this for S'/B'/B0
        bakpct = 1.0 - (totevents[1][i][j][k]/maxtot[1])

        if (abs(sigpct)<=2.1 and abs(bakpct)<=2.1):
          if g<6:
            tempypts[i].append(sigpct)
            tempxpts[i].append(bakpct)
            #print str(i) + " " + str(g) + " " + str(sigpct) + " " + str(bakpct)
          elif g>=6 and g<12:
            tempypts[6+j].append(sigpct)
            tempxpts[6+j].append(bakpct)
          elif g>=12 and g<18:
            tempypts[12+k].append(sigpct)
            tempxpts[12+k].append(bakpct)
          ################
          # Fill the histos...but only once
          ################
          if g==0:
            hsig[i].Fill(xname, sigpct)
            hbak[i].Fill(xname, 1.0-bakpct)
            hsig[6+j].Fill(xname, sigpct)
            hbak[6+j].Fill(xname, 1.0-bakpct)
            hsig[12+k].Fill(xname, sigpct)
            hbak[12+k].Fill(xname, 1.0-bakpct)

  ###########################
  # grroc
  ###########################
  #print "len: " + str(len(tempxpts[g]))
  grroc.append(TGraph(len(tempxpts[g]), tempxpts[g], tempypts[g]))
  color = g%6 + 1
  if color == 5:
    color = 22
  grroc[g].SetMarkerColor(color)
  grroc[g].SetMarkerStyle(20)
  grroc[g].SetMarkerSize(1.0)
  
  grroc[g].SetTitle()

  grroc[g].GetYaxis().SetNdivisions(6)
  grroc[g].GetYaxis().CenterTitle()
  grroc[g].GetYaxis().SetLabelSize(0.06)
  grroc[g].GetYaxis().SetTitleSize(0.07)
  grroc[g].GetYaxis().SetTitleOffset(0.6)
  grroc[g].GetYaxis().SetTitle("% signal retained")

  grroc[g].GetXaxis().SetNdivisions(8)
  grroc[g].GetXaxis().CenterTitle()
  grroc[g].GetXaxis().SetLabelSize(0.06)
  grroc[g].GetXaxis().SetTitleSize(0.09)
  grroc[g].GetXaxis().SetTitleOffset(1.0)
  #grroc[g].GetXaxis().SetTitle(tag + ": % background rejection")
  grroc[g].GetXaxis().SetTitle("% background rejection")

  ###########################
  # hsig
  ###########################
  color = g%6 + 1
  if color == 5:
    color = 22
  hsig[g].SetMarkerColor(color)
  hsig[g].SetFillColor(color)
  hsig[g].SetMarkerStyle(23)
  hsig[g].SetMarkerSize(1.0)
  
  hsig[g].SetTitle("")
  hsig[g].SetMaximum(1.4) 
  hsig[g].SetMinimum(0.0)

  hsig[g].GetYaxis().SetNdivisions(6)
  hsig[g].GetYaxis().CenterTitle()
  hsig[g].GetYaxis().SetLabelSize(0.06)
  hsig[g].GetYaxis().SetTitleSize(0.07)
  hsig[g].GetYaxis().SetTitleOffset(0.6)
  hsig[g].GetYaxis().SetTitle("% efficiency")

  hsig[g].GetXaxis().SetNdivisions(8)
  hsig[g].GetXaxis().CenterTitle()
  hsig[g].GetXaxis().SetLabelSize(0.06)
  hsig[g].GetXaxis().SetTitleSize(0.09)
  hsig[g].GetXaxis().SetTitleOffset(1.0)
  #hsig[g].GetXaxis().SetTitle(tag + ": Selector combo")
  hsig[g].GetXaxis().SetTitle("Selector combo")
  hsig[g].GetXaxis().LabelsOption("v")

  ###########################
  # hbak
  ###########################
  hbak[g].SetMarkerColor(color)
  hbak[g].SetFillColor(color)
  hbak[g].SetMarkerStyle(22)
  hbak[g].SetMarkerSize(1.0)
  
  hbak[g].SetTitle("")

  hbak[g].GetYaxis().SetNdivisions(6)
  hbak[g].GetYaxis().CenterTitle()
  hbak[g].GetYaxis().SetLabelSize(0.06)
  hbak[g].GetYaxis().SetTitleSize(0.07)
  hbak[g].GetYaxis().SetTitleOffset(0.6)
  hbak[g].GetYaxis().SetTitle("% efficiency")

  hbak[g].GetXaxis().SetNdivisions(8)
  hbak[g].GetXaxis().CenterTitle()
  hbak[g].GetXaxis().SetLabelSize(0.06)
  hbak[g].GetXaxis().SetTitleSize(0.09)
  hbak[g].GetXaxis().SetTitleOffset(1.0)
  hbak[g].GetXaxis().SetTitle("Selector combo")
  hbak[g].GetXaxis().LabelsOption("v")

##################################
# Plot the ROC graphs
##################################
gcan = TCanvas("gcan","gcan",200,200,1200,1000)
gcan.SetFillColor(0)
gcan.Divide(1,3)

legend = []
legcount = -1

for g in range(0,18):
  gcan.cd((g/6)+1)
  grroc[g].GetXaxis().SetLimits(0.3,1.2)
  grroc[g].GetYaxis().SetRangeUser(0.4,1.1)
  if g%6 == 0:
    grroc[g].Draw("ap")
    #print str(g) + " plotting first..."
  else:
    grroc[g].Draw("p")
    #print str(g) + " plotting not first..."

  ##############################
  # Make some legends
  ##############################
  if g%6 == 0:
    legend.append(TLegend(0.7, 0.5, 0.99, 0.99))
    legcount += 1 

  if g<6:
    legend[legcount].AddEntry(grroc[g], pselectors[g], "p")
  elif g>=6 and g<12:
    legend[legcount].AddEntry(grroc[g], kselectors[g-6], "p")
  elif g>=12 and g<18:
    legend[legcount].AddEntry(grroc[g], piselectors[g-12], "p")

  legend[legcount].Draw()

  gcan.Update()



sigbaklegend = []
##################################
# Plot the sig/bak graphs
##################################
hsbcans = []
for i in range(0,4):
  if i!=3:
    hsbcans.append(TCanvas("hsbcans"+str(i),"",250+10*i,250+10*i,1200,1000))
    hsbcans[i].SetFillColor(0)
    hsbcans[i].Divide(2,3)
  else:
    hsbcans.append(TCanvas("hsbcans"+str(i),"",250+10*i,250+10*i,1200, 500))
    hsbcans[i].SetFillColor(0)
    hsbcans[i].Divide(1,1)

legcount = -1
firstTime = 0
allLeg = TLegend(0.8, 0.3, 0.99, 0.99)
for g in range(0,18):
  if g<6:
    sigbaklegend.append(TLegend(0.3, 0.8, 0.99, 0.99, pselectors[g]))
  elif g>=6 and g<12:
    sigbaklegend.append(TLegend(0.3, 0.8, 0.99, 0.99, kselectors[g-6]))
  elif g>=12:
    sigbaklegend.append(TLegend(0.3, 0.8, 0.99, 0.99, piselectors[g-12]))
  legcount += 1 
  #print g/6
  hsbcans[g/6].cd((g%6)+1)
  #print "legcount: " + str(legcount)
  sigbaklegend[legcount].AddEntry(hsig[g], "Signal eff/survival", "p")
  sigbaklegend[legcount].AddEntry(hbak[g], "Background eff/survival", "p")
  #hsbc[g].GetXaxis().SetLimits(0.3,1.0)
  #grroc[g].GetYaxis().SetRangeUser(0.4,1.0)
  hsig[g].Draw("p")
  hbak[g].Draw("psame")

  sigbaklegend[legcount].Draw()

  hsbcans[g/6].Update()

  if g<6:
    hsbcans[3].cd(1)
    gPad.SetRightMargin(0.21)
    if firstTime == 0:
      hsig[g].Draw("p")
      hbak[g].Draw("psame")
      firstTime = 1
    else:
      hsig[g].Draw("psame")
      hbak[g].Draw("psame")
      #print "here"
    allLeg.AddEntry(hsig[g], pselectors[g], "f")
    allLeg.Draw()
    gPad.Update()

##################################
# Fill with some points
##################################
bestsig = 0.0
worstsig = 1e9
bestp = 0
bestpi = 0
bestk = 0
a = 5.0

for g in range(0,18):
  for i in range(0,6):
    for j in range(0,6):
      for k in range(0,6):
        xname = str(i) + str(j) + str(k)

        sigpct = sigevents[0][i][j][k]/maxsig[0]
        #bakpct = (bakevents[1][i][j][k]/maxbak[1])
        # Use this for S'/B'/B0
        bakpct = (totevents[1][i][j][k]/maxtot[1])

        # Calculate the significance
        significance = 0.0
        fomtype = ""
        #print use_sprime
        if whichsig == "nosig" and use_sprime == False:
          #significance = numsig*sigpct/sqrt(numbak*bakpct)
          significance = sigpct/sqrt(bakpct)
          fomtype = "sqrt(s)/sqrt(b)"
        elif use_sprime == True:
          significance = sigpct/sqrt(numsig*sigpct + numbak*bakpct)
          fomtype = "sqrt(s)/sqrt(s+b)"
          print "Using sprimes.............................."
        else:
          fomtype = "Punzi"
          #significance = numsig*sigpct/sqrt(numsig*sigpct + numbak*bakpct)
          # Try Punzi
          #print "Punzi"
          significance = sigpct/(sqrt(numbak*bakpct) + a/2.0)

        # Catch bad fits
        if sigpct > 1.1 or bakpct > 1.1 or sigpct<0.10 or bakpct<0.10:
          significance = 0.0

        #print "%d %d %f %f" % ( i, j, significance, bestsig  )
        if significance > bestsig:
          print fomtype
          if 1:
            bestsig = significance
            bestp = i
            bestpi = j
            #print "Found a new best"
            #print " & %d & %d & %d & %d & %3.4f & %3.4f & %3.4f \\\\" % ( lepsel, i, j, k, bestsig, sigpct, bakpct )
            print " & %d & %d & %d & %3.4f & %3.4f & %3.4f \\\\" % ( i, j, k, bestsig, sigpct, bakpct )
        if significance < worstsig:
          worstsig = significance


        #if (abs(sigpct)<=2.1 and abs(bakpct)<=2.1):
        if (1):
          ################
          # Fill the histos...but only once
          ################
          if g==0:
            hsignif[i].Fill(xname, significance)

hmax = 0
hmin = 1e9
for g in range(0,6):
  #hm = hsignif[g].GetBinContent(3)
  hm = hsignif[g].GetMaximum()
  #print hm
  if hm > hmax:
    hmax = hm
  hm = hsignif[g].GetMinimum()
  if hm < hmin:
    hmin = hm

#print hmax
#print hmin

for g in range(0,18):
  #  if g<6:
  #hsignif[g].SetMarkerColor(g/36 + 1)
  #elif g>=6 and g<12:
  #hsignif[g].SetMarkerColor(g/6 + 1)
  #else:
  color = g%6 + 1
  if color == 5:
    color = 22
  hsignif[g].SetMarkerColor(color)
  hsignif[g].SetMarkerStyle(20)
  hsignif[g].SetMarkerSize(1.0)
  
  hsignif[g].SetTitle("")

  #hsignif[g].SetMinimum(0.5)
  #hsignif[g].SetMaximum(1.8)

  #print worstsig
  #print bestsig
  hsignif[g].SetMinimum(0.8*worstsig)
  hsignif[g].SetMaximum(1.2*bestsig)



  #hsignif[g].SetMinimum(hmin - 0.05*(hmax-hmin))
  #hsignif[g].SetMaximum(hmax + 0.10*(hmax-hmin))

  hsignif[g].GetYaxis().SetNdivisions(6)
  hsignif[g].GetYaxis().CenterTitle()
  hsignif[g].GetYaxis().SetLabelSize(0.06)
  hsignif[g].GetYaxis().SetTitleSize(0.07)
  hsignif[g].GetYaxis().SetTitleOffset(0.6)
  hsignif[g].GetYaxis().SetTitle("Significance")

  hsignif[g].GetXaxis().SetNdivisions(8)
  hsignif[g].GetXaxis().CenterTitle()
  hsignif[g].GetXaxis().SetLabelSize(0.06)
  hsignif[g].GetXaxis().SetTitleSize(0.09)
  hsignif[g].GetXaxis().SetTitleOffset(1.0)
  #hsignif[g].GetXaxis().SetTitle(tag + ": Selector combo")
  hsignif[g].GetXaxis().SetTitle("Selector combo")
  hsignif[g].GetXaxis().LabelsOption("v")

##################################
# Plot the ROC graphs
##################################
cansig = []

for i in range(0,2):
  if i==0:
    cansig.append(TCanvas("cansig"+str(i),"",400+10*i,400+10*i,1200,1000))
    cansig[i].SetFillColor(0)
    cansig[i].Divide(2,3)
  else:
    cansig.append(TCanvas("cansig"+str(i),"",400+10*i,400+10*i,1200, 500))
    cansig[i].SetFillColor(0)
    cansig[i].Divide(1,1)

legsignif = []
legcount = -1
allLegSig = TLegend(0.8, 0.3, 0.99, 0.99)
for g in range(0,6):
  cansig[0].cd(g+1)
  legcount += 1
  #hsignif[g].GetXaxis().SetLimits(6*g, 6*(g+1))
  hsignif[g].Draw("p")
  #sigbaklegend.append(TLegend(0.3, 0.8, 0.99, 0.99, pselectors[g]))
  legsignif.append(TLegend(0.3, 0.9, 0.99, 0.99))
  legsignif[legcount].AddEntry(hsignif[g], pselectors[g], "p")
  legsignif[legcount].Draw()
  gPad.Update()

  cansig[1].cd(1)
  gPad.SetRightMargin(0.21)

  if g==0:
    #print whichsig
    if whichsig == "nosig" and use_sprime == False:
      allLegSig.AddEntry(hsignif[g], "S/#sqrt{B}", "p")
    elif use_sprime == True:
      name = "S/#sqrt{S + B}, S=%d and B=%d  " % (numsig, numbak)
      allLegSig.AddEntry(hsignif[g], name, "p")
    else:
      # Punzi
      name = "Punzi: #epsilon_{S}/(#sqrt{B#epsilon_{B}} + a/2), B=%d and a=%d  " % (numbak, a)
      allLegSig.AddEntry(hsignif[g], name, "p")

  allLegSig.AddEntry(hsig[g], pselectors[g], "f")
  allLegSig.Draw()
  if g==0:
    hsignif[g].Draw("p")
  else:
    hsignif[g].Draw("psame")


####################################
# Save the plots
####################################
for j in range(0,1):
  name = "Plots/combined_sigbkg_selectors_roc_" + tag + "_" + str(j) + ".eps" 
  gcan.SaveAs(name)
for j in range(0,4):
  name = "Plots/combined_sigbkg_selectors_effs_" + tag + "_" + str(j) + ".eps" 
  hsbcans[j].SaveAs(name)
for j in range(0,2):
  name = "Plots/combined_sigbkg_selectors_significance_" + tag + "_" + str(j) + ".eps" 
  cansig[j].SaveAs(name)


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
