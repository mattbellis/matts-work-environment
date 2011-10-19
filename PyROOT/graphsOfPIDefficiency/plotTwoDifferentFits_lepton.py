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
tag = sys.argv[5]

plotMax = 6
if sys.argv[1].find("mu") >= 0:
  plotMax = 8

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
gStyle.SetPadLeftMargin(0.15)
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
  print infiles[f]
  file = open(infiles[f])
  for line in file:
    lines[f].append(line)

tot = []
tot_after = []
pct = []

for f in range(0,2):
  tot.append([])
  tot_after.append([])
  pct.append([])

##########################################
#
#
##########################################
legcount = 0
count = 0
hmax = 1.0
for f in range(0,2):
  linecount = 0
  print infiles[f]
  if 1:
    #print f
    ################################
    # Get the first histo
    ################################
    for k in range(0,plotMax):
      t = float(lines[f][linecount].split()[1])
      t_a = float(lines[f][linecount].split()[2])
      tot[f].append(t)
      tot_after[f].append(t_a)
      pct[f].append(t_a/t)
      linecount += 1

xpts = array('f')
ypts = array('f')

hsig = []
hbak = []
hsignif = []
grroc = [] 
tempxpts = []
tempypts = []

func = []

hdum = []

vary = [-0.00, 0.00, 0.00]

for g in range(0,3*plotMax):
  name = "func" + str(g)
  index= g/3
  sigpct = str(pct[0][index] + (pct[0][index] * vary[g%3]))
  # This is what I initially had
  #function = "(" + sigpct + ")/sqrt( (" + str(tot_after[1][index]) + " - x*" + sigpct + ") / (" + str(tot[0][index]) + " - x) )"

  # Me fixed? YES!!!!
  function = "(" + sigpct + ")*sqrt( (" + str(tot[1][index]) + " - x) / (" + str(tot_after[1][index]) + " - (x * " + sigpct + ")) )"
  # This is suggested by Elliot
  #function = "(" + sigpct + ")* (x)/ sqrt(" + str(tot_after[1][index]) + " )"
  print function
  func.append(TF1(name,function, 0 , tot[1][index]) )
  func[g].SetLineColor(2 + 2*(index%4))
  func[g].SetLineWidth(5)
  func[g].SetLineStyle(1*(index/4) + 1)

  hname = "hdum" + str(g)
  hdum.append(TH1F(hname,hname,10,0,1.4*tot[1][index]))
  hdum[g].SetMinimum(0)
  hdum[g].SetTitle("")
  hdum[g].GetYaxis().CenterTitle()
  hdum[g].GetYaxis().SetNdivisions(4)
  hdum[g].GetYaxis().SetLabelSize(0.06)
  hdum[g].GetYaxis().SetTitleOffset(0.7)
  hdum[g].GetYaxis().SetTitleSize(0.09)
  hdum[g].GetYaxis().SetTitle("S / #sqrt{B}")

  hdum[g].GetXaxis().SetNdivisions(6)
  hdum[g].GetXaxis().CenterTitle()
  hdum[g].GetXaxis().SetLabelSize(0.06)
  hdum[g].GetXaxis().SetTitleSize(0.09)
  hdum[g].GetXaxis().SetTitleOffset(1.0)
  hdum[g].GetXaxis().SetTitle("# of assumed true events")

  hdum[g].SetFillStyle(0)

#exit(-1)

##################################
# Plot the ROC graphs
##################################
gcan = TCanvas("gcan","gcan",10,10,1200,1000)
gcan.SetFillColor(0)
gcan.Divide(1,3)

cantalk = []
for g in range(0,10):
  cantalk.append(TCanvas("cantalk"+str(g),"cantalk"+str(g),20+10*g,20+10*g,1200,800))
  cantalk[g].SetFillColor(0)
  cantalk[g].Divide(1,1)

legend = []
legcount = -1
talklegend = []
talklegcount = -1

for g in range(0,plotMax):
  gcan.cd(1)
  if g==0:
    hdum[0].SetMaximum(300)
    hdum[0].Draw()
  func[3*g+0].DrawCopy("same")
  #func[3*g+1].DrawCopy("same")
  #func[3*g+2].DrawCopy("same")
  if g == 0:
    legend.append(TLegend(0.6, 0.5, 0.99, 0.99))
  legend[0].AddEntry(func[3*g], lep_selectors[g], "l")
  legend[0].Draw()

  # Zoom in 
  gcan.cd(2)
  if g==0:
    hdum[1].SetMinimum(1.5)
    hdum[1].SetMaximum(5.0)
    hdum[1].Draw()
  func[3*g+0].DrawCopy("same")
  #func[3*g+1].DrawCopy("same")
  #func[3*g+2].DrawCopy("same")
  if g == 0:
    legend.append(TLegend(0.6, 0.5, 0.99, 0.99))
  legend[1].AddEntry(func[3*g], lep_selectors[g], "l")
  legend[1].Draw()

  # Zoom in more
  gcan.cd(3)
  if g==0:
    hdum[2].SetMinimum(2.0)
    hdum[2].SetMaximum(3.0)
    hdum[2].GetXaxis().SetLimits(0.0, 15000)
    hdum[2].Draw()
  func[3*g+0].DrawCopy("same")
  #func[3*g+1].DrawCopy("same")
  #func[3*g+2].DrawCopy("same")
  if g == 0:
    legend.append(TLegend(0.6, 0.5, 0.99, 0.99))
  legend[2].AddEntry(func[3*g], lep_selectors[g], "l")
  legend[2].Draw()

  gcan.Update()
  ##############################
  # Make some legends
  ##############################

for i in range(0,plotMax+2):
  print i
  cantalk[i].cd(1)
  if i<plotMax:
    hdum[0].Draw()
  elif i==plotMax:
    hdum[1].Draw()
  elif i==plotMax+1:
    hdum[2].Draw()

  maxplot = i+1
  if i>=plotMax:
    maxplot = plotMax
  for g in range(0,maxplot):
    func[3*g+1].DrawCopy("same")
    if g==0:
      talklegend.append(TLegend(0.6, 0.5, 0.99, 0.99))
    talklegend[i].AddEntry(func[3*g], lep_selectors[g], "l")
  talklegend[i].Draw()

  cantalk[i].Update()
##############################


####################################
# Save the plots
####################################
for j in range(0,1):
  name = "plots/mu_combined_sigbkg_selectors_roc_" + tag + "_" + str(j) + "_" + ".eps" 
  gcan.SaveAs(name)
for i in range(0,10):
  name = "plots/mu_combined_sigbkg_selector_curves_" + tag + "_" + str(i) + "_" + ".eps" 
  cantalk[i].SaveAs(name)

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
