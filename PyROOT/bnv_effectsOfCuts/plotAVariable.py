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
if len(sys.argv)>2:
  doTalkPlots = bool(sys.argv[2])

if htype == "hmass":
  nvars = 3
elif htype == "hshape":
  nvars = 32
elif htype == "h2D":
  nvars = 2


index = []
if len(sys.argv)>=7:
  index = [ int(sys.argv[5]), int(sys.argv[6]) ]

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True


gROOT.Reset()
#gStyle.SetOptStat(10)
gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadBottomMargin(0.20)
#gStyle.SetPalette(1)
set_palette("palette",100)


canvastitles = ["B", "B","B"]

canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"
canvastitles[1] = "B^{0} #rightarrow #Lambda_{C}^{-} #mu^{+}"
canvastitles[2] = "B^{0} #rightarrow #Lambda_{C} #mu"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,nvars*ncuts):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 1200, 750 ))
  #can.append(TCanvas( name, name, 20*(f/ncuts)+10*(f%ncuts), 10+10*(f%ncuts), 200, 150 ))
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
  bottompad[f].Divide(3, 3);

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(canvastitles[f])
  canvastext[f].AddText("Stuff")
  canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()


#
# Single canvases
#
if doTalkPlots:
  cansingle = []
  for f in range(0,9):
    cansingle.append([])
    for i in range(0,1):
      cansingle[f].append([])
      for j in range(0,nvars):
        cansingle[f][i].append([])
        for k in range(0,ncuts):
          name = "cansingle" + str(f) + "_" + str(i) + "_" + str(j) + "_" + str(k)
          index = k + j*ncuts
          cansingle[f][i][j].append(TCanvas( name, name, (20*f)+10*(index%(ncuts*nvars)), 10+10*(index%(ncuts*nvars)), 150, 150 ))
          cansingle[f][i][j][k].SetFillColor( 0 )
          cansingle[f][i][j][k].Divide( 1,1 )
          cansingle[f][i][j][k].cd(1)


hstack = [[], []]
histos = []
text1 = []
for f in range(0,8):
  histos.append([])
  text1.append([])
  for i in range(0,1):
    histos[f].append([])
    text1[f].append([])
    for j in range(0,32):
      histos[f][i].append([])
      text1[f][i].append([])
      for k in range(0,16):
        histos[f][i][j].append([])
        text1[f][i][j].append([])

for i in range(0,1):
  hstack[0].append([])
  hstack[1].append([])
  for j in range(0,32):
    hstack[0][i].append([])
    hstack[1][i].append([])
    for k in range(0,16):
      hstack[0][i][j].append([])
      hstack[1][i][j].append([])

datasets = ["SP9446", "OffPeak", "OnPeak", "SP1005", "SP998", "SP1235", "SP1237"]
datasettext = ["Signal1", "Off peak", "On Peak (blind)", "c#bar{c}", "u#bar{u}/d#bar{d}/s#bar{s}", "B^{+}B^{-} generic", "B^{0}#bar{B}^{0} generic"] 
colors = [2, 6, 4, 23, 26, 30, 36, 4, 6]
scale_amount = [1.0, 10.0, 1.0, 0.5, 0.5, 0.35, 0.35]

#
# Open a ROOT file and save the formula, function and histogram
#
numfiles = len(datasets)
filename = []
for f in range(0,numfiles):
  name = "rootFiles/" + htype+"_"+datasets[f] + "_nocut_mediumErange_pids_Lam_fitErange_pids_Lam_4TMVA" + ".root"
  filename.append(name)

rootfile = []
for f in range(0,numfiles):
  print filename[f]
  rootfile.append(TFile( filename[f] ))
  if os.path.isfile(filename[f] ):
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
                histos[f][i][j][k] = TH2F(hname,hname,10,0,1)

            hdum = histos[f][i][j][k].Clone()
          # Draw the canvas labels
          
          bottompad[k + ncuts*j].cd(f + 1)

          #histos[f][i][j][k].Sumw2()

          #if(j==2):
            #histos[f][i][j][k].Rebin(3)
          #else:
          if htype != "h2D":
            histos[f][i][j][k].Rebin(2)
          histos[f][i][j][k].Scale(scale_amount[f])
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

          if f<3:
            histos[f][i][j][k].SetLineWidth(2)

          histos[f][i][j][k].SetFillColor(colors[f])
          
          if htype != "h2D":
            #print str(f)+" "+str(i)+" "+str(j)+" "+str(k)
            ##print histos[f][i][j][k]
            histos[f][i][j][k].DrawCopy()
            histos[f][i][j][k].DrawCopy("samee")
          else:
            histos[f][i][j][k].DrawCopy("colz")


          bottompad[k + ncuts*j].cd(f + 1)
          text1[f][i][j][k] = TPaveText(0.0,0.8,0.4,1.0,"NDC")
          text1[f][i][j][k].AddText(datasettext[f])
          text1[f][i][j][k].SetBorderSize(1)
          text1[f][i][j][k].SetFillStyle(1)
          text1[f][i][j][k].Draw()

          if k==ncuts-1 and j==1:
            print "cont: " + str(f) + " " + str(i) + "_" + str(j) + "_" + str(k)
            print histos[f][i][j][k].Integral(35,50) 

          if f == 0:
            name = "hstack0_" + str(i) + "_" + str(j) + "_" + str(k)
            hstack[0][i][j][k] = THStack(name,name)
            name = "hstack1_" + str(i) + "_" + str(j) + "_" + str(k)
            #print name
            hstack[1][i][j][k] = THStack(name,name)
          if f >=3 and f<5:
            hstack[0][i][j][k].Add( histos[f][i][j][k].Clone() )
            bottompad[k + ncuts*j].cd(8)
            if f==4:
              if htype!="h2D":
                histos[1][i][j][k].Draw("e")
                hstack[0][i][j][k].Draw("same")
                histos[1][i][j][k].Draw("samee")
              else:
                hstack[0][i][j][k].Draw("colz")
          if f >=3:
            hstack[1][i][j][k].Add( histos[f][i][j][k].Clone() )
            bottompad[k + ncuts*j].cd(9)
            if f==6:
              if htype!="h2D":
                histos[2][i][j][k].Draw("e")
                hstack[1][i][j][k].Draw("same")
                histos[2][i][j][k].Draw("samee")
              else:
                hstack[1][i][j][k].Draw("colz")

          can[k+ncuts*j].Update()

          ###########################
          # Talk plots 
          ###########################
          if doTalkPlots:
            cansingle[f][i][j][k].cd(1)
            if histos[f][i][j][k]:
              if htype != "h2D":
                histos[f][i][j][k].DrawCopy()
              else:
                histos[f][i][j][k].DrawCopy("colz")

            if f==4:
              cansingle[7][i][j][k].cd(1)
              if htype!="h2D":
                histos[1][i][j][k].Draw("e")
                hstack[0][i][j][k].Draw("same")
                histos[1][i][j][k].Draw("samee")
              else:
                hstack[0][i][j][k].Draw("colz")
            if f==6:
              cansingle[8][i][j][k].cd(1)
              if htype!="h2D":
                histos[2][i][j][k].Draw("e")
                hstack[1][i][j][k].Draw("same")
                histos[2][i][j][k].Draw("samee")
              else:
                hstack[1][i][j][k].Draw("colz")

            cansingle[f][i][j][k].Update()
          ###########################


whichPlot = ""
baryon = ""
ntp = ""
#if len(index)==2:
  #for f in range(0,5):
    #name = "Plots/cansingle" + str(f) + "_" +str(index[0])+"_"+str(index[1])+"_" + baryon + "_" + ntp + "_" + htype + "_" + str(whichPlot) + ".eps" 
    #cansingle[f].SaveAs(name)


for j in range(0,nvars):
  name = "Plots/can" + str(j) + "_" + baryon + "_" + ntp + "_" + htype + "_" + str(whichPlot) + ".ps" 
  can[j].SaveAs(name)

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
