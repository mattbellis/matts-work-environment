#!/usr/bin/env python
#
#


# Import the needed modules
import os
import sys

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import gROOT, gStyle, gPad, TLegend, TLine
import ROOT

import math

from color_palette import *


batchMode = False

offset = 0
doSingles = 0
doLogz = 0

## Set which topology is the raw
#rawtop = 0 # ppippim
rawtop = 3 # pippim_p

#
# Parse the command line options
#
fit = sys.argv[1]
lomin = int(sys.argv[2])
himax = int(sys.argv[3])
chi2cut = float(sys.argv[4])
tag = sys.argv[5]
offset = int(sys.argv[6])
doSingles = int(sys.argv[7])
doLogz = int(sys.argv[8])


#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
# Display no statisitics on the histograms
gStyle.SetOptStat(0)
# Set some drawing styles
gStyle.SetLabelSize(0.05,"X")
#gStyle.SetPalette(1)
set_palette("palette",100)

whichtopology="ppippim"


###################
kv = ["ppipM", "ppimM", "pippimM", "pCosThetaCM", "pipCosThetaCM", "pimCosThetaCM",
"pipCosThetaDpHel", "pipPhiDpHel", 
"pimCosThetaD0Hel", "pimPhiD0Hel", 
"pipCosThetaRhoHel", "pipPhiRhoHel"]



type = ["data","acc","acc_wt"]
topology = ["ppippim", "ppip_pim", "ppim_pip", "pippim_p"]
topology_latex = ["p#pi^{+}#pi^{-}", "p#pi^{+}(#pi^{-})", "p(#pi^{+})#pi^{-}", "(p)#pi^{+}#pi^{-}", "d#sigma / dX", "d#sigma /(dX dY)", "d#sigma /(dX dY)"]

#
# Draw the pads and graphs
#

name = ""
title = ""
xtitle = ""
filename = ""

# Info about how much to plot
numiter = 5;
numcan = 16 + 4 + 5; # data/acc + raw + raw2d

#
# Open output file for what is good and what is not
name = "goodFits_chi2cut%2.2f_%d-%d.txt" % (chi2cut,lomin,himax)
OUT = open(name, "w");
#
print "About to grab the histos.........."
#
bestlogL = [ 1e12, 1e12, 1e12, 1e12, 1e12,  1e12, 1e12, 1e12, 1e12, 1e12 ] 
bestchi2 = [ 1e12, 1e12, 1e12, 1e12, 1e12,  1e12, 1e12, 1e12, 1e12, 1e12 ] 

x = 0.0
logLdum = 0.0

wbinlo = []
wbinhi = []

# Declare the histograms
h = []
h2d = []
errors = []

inFile = []
gr_logL = []
bestlogL = []
logL = []
foundAtLeastOne = False
numPlots = 0
for lo in range(lomin, himax, 10):
  hi = 10 + lo
  
  # Try to open the root file
  filename = "rootFiles/%s_%d-%d.root" % (fit, lo, hi)
  inFile.append(TFile(filename))
  foundAtLeastOne = False

  h.append([])
  h2d.append([])
  errors.append([])

  # Is it a good file?
  if not inFile[numPlots].IsZombie():
    print "Able to open file: " + filename
    #
    # Grab the logL iter info 
    #
    gr_logL.append(inFile[numPlots].Get("gr_logL"))
    bestlogL.append(1e6)
    logL.append([])

    #
    # Loop over the different fit iterations
    #
    for n in range(0, numiter):
      logLdum = ROOT.Double(0.0)
      x = ROOT.Double(0.0)
      gr_logL[numPlots].GetPoint(n,x,logLdum)

      if(logLdum < bestlogL[numPlots]):
        bestlogL[numPlots] = logLdum
        #print "bestLogL " + str(bestlogL[numPlots])
        logL[numPlots].append(logLdum)

    wbinlo.append(lo)
    wbinhi.append(hi)



    #
    # Grab the histograms
    #
    for i in range(0,5): # w bins
      h[numPlots].append([])
      h2d[numPlots].append([])
      errors[numPlots].append([])
      for j in range(0,4): # topologies
        h[numPlots][i].append([])
        h2d[numPlots][i].append([])
        errors[numPlots][i].append([])
        for k in range(0,15): # kinvars
          h[numPlots][i][j].append([])
          h2d[numPlots][i][j].append([])
          errors[numPlots][i][j].append([])
          for n in range(0,5): # iterations, fill the histogram array first
            h[numPlots][i][j][k].append([])
            h2d[numPlots][i][j][k].append([])
            errors[numPlots][i][j][k].append([])
          for n in range(0,5): # iterations
            N=n # iteration
            J=j # iteration
            if i==0 or i==2 or i==4:
              N=0
            if i==4 or i==5:
              J=0
            #
            # Grab the histograms
            #
            # 1D
            name = "h_%d_%d_%d_%d" % (i,J,k,N)
            hdum = inFile[numPlots].Get(name)
            if hdum:
              #print "Found: " + name 
              h[numPlots][i][J][k][N] = hdum
              h[numPlots][i][J][k][N].SetName(name)
            else:
              h[numPlots][i][J][k][N] = TH1F()

            # 2D
            name = "h2d_%d_%d_%d_%d" % (i,J,k,N)
            h2ddum = inFile[numPlots].Get(name)
            if h2ddum:
              #print "Found: " + name 
              h2d[numPlots][i][J][k][N] = h2ddum
              h2d[numPlots][i][J][k][N].SetName(name)
              h2d[numPlots][i][J][k][N].GetYaxis().SetLabelSize(0.05)
              h2d[numPlots][i][J][k][N].GetYaxis().SetTitleSize(0.075)
              h2d[numPlots][i][J][k][N].GetYaxis().SetTitleFont(42)
              h2d[numPlots][i][J][k][N].GetYaxis().SetTitleOffset(1.1)
              h2d[numPlots][i][J][k][N].GetYaxis().CenterTitle()
              h2d[numPlots][i][J][k][N].GetXaxis().SetLabelSize(0.05)
              h2d[numPlots][i][J][k][N].GetXaxis().SetTitleSize(0.075)
              h2d[numPlots][i][J][k][N].GetXaxis().SetTitleFont(42)
              h2d[numPlots][i][J][k][N].GetXaxis().SetTitleOffset(1.1)
              h2d[numPlots][i][J][k][N].GetXaxis().CenterTitle()

        foundAtLeastOne=True;

    numPlots+=1
    # Closing the loop that looped over the files
  

  print "NUMPLOTS: " + str(numPlots) 

#
# Make the canvases
#
can = []
top = []
bottom = []
l1 = []
# Get the dimensions
rows = int(math.sqrt(numPlots))
cols = numPlots/rows + 1
#
for i in range(0,numcan):
  # make the canvases and top and bottom pads
  name = "can%d" % i
  title = "Data %d" % i
  if numPlots<=5:
    can.append(TCanvas(name, title, 10 + 10 * i, 10 + 10 * (i%4), 280*numPlots, 900))
  else:
    can.append(TCanvas(name, title, 10 + 10 * i, 10 + 10 * (i%4), 280*5, 900))
  can[i].SetFillColor(0)
  top.append(TPad("top", "The Top", 0.01, 0.85, 0.99, 0.99))
  top[i].SetFillColor(0)
  top[i].Draw()
  bottom.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.85))
  bottom[i].SetFillColor(0)
  bottom[i].Draw()
  bottom[i].Divide(numPlots, 4)


  # Titles
  top[i].cd();
  title = "Fit: %s %s " % (fit, topology_latex[i/4])
  l1.append(TPaveLabel(0.01, 0.01, 0.99, 0.99, title))
  l1[i].SetFillStyle(1)
  l1[i].SetFillColor(1)
  l1[i].SetTextColor(0)
  l1[i].Draw()

#sys.exit(-1)

fillcolor = [42, 0, 0]
linewidth = [1, 2, 2]
linecolor = [1, 4, 2]

axisScale = 1.3;

tpl_wbin = []

############################################
# Print out all the histograms
############################################
for f in range(0,numPlots): # w bins
  tpl_wbin.append([])
  for i in range(0,5): # types of plots data/accwt/acc/rawwt/raw
    for j in range(0,4): # topologies
      for k in range(0,12): # kinvars
        for n in range(0,5): # iterations
          #
          # Figure out where to display the histogram
          #
          whichcan = 0
          whichpad = 0
          if i<3:
            whichcan = k/3 + j*4
            whichpad = f + numPlots*(k%3) + numPlots+1
          else: 
            whichcan = k/3 + 16
            whichpad = f + numPlots*(k%3) + numPlots+1

          #print "whichcan/whichpad: " + str(whichcan) + " " + str(whichpad)
          bottom[whichcan].cd(whichpad)
          gPad.SetBottomMargin(0.18)
          gPad.SetLeftMargin(0.18)

          # Format the hisograms
          #print str(f) + " " +  str(i) + " " +  str(j) + " " +  str(k) + " " +  str(n)
          # Break for the ones which only have one or two plots
          if h[f][i][j][k][n] and \
              ( i==1 or i==3 or \
              ((i==0 or i==2 ) and n==0 ) or \
              ( i==4 and n==0 and j==rawtop ) or \
              ( i==5 and j==rawtop) ):
            #print str(f) + " " +  str(i) + " " +  str(j) + " " +  str(k) + " " +  str(n)
            h[f][i][j][k][n].Rebin(4)
            h[f][i][j][k][n].SetMinimum(0)
            h[f][i][j][k][n].SetTitle("")
            h[f][i][j][k][n].GetXaxis().SetNdivisions(6)
            h[f][i][j][k][n].GetYaxis().SetTitleFont(42)
            h[f][i][j][k][n].GetYaxis().SetTitleSize(0.075)
            h[f][i][j][k][n].GetYaxis().SetTitleOffset(1.1)
            h[f][i][j][k][n].GetYaxis().CenterTitle()
            h[f][i][j][k][n].GetXaxis().SetLabelSize(0.05)
            h[f][i][j][k][n].GetXaxis().SetTitleSize(0.075)
            h[f][i][j][k][n].GetXaxis().SetTitleFont(42)
            h[f][i][j][k][n].GetXaxis().SetTitleOffset(1.1)
            h[f][i][j][k][n].GetXaxis().CenterTitle()
            if(k==0):
              xtitle = "M(p #pi^{+}) GeV/c^{2}"
            elif(k==1):
              xtitle = "M(p #pi^{-}) GeV/c^{2}"
            elif(k==2):
              xtitle = "M(#pi^{+} #pi^{-}) GeV/c^{2}"
            elif(k==3):
              xtitle = "p cos(#theta) CM"
            elif(k==4):
              xtitle = "#pi^{+} cos(#theta) CM"
            elif(k==5):
              xtitle = "#pi^{-} cos(#theta) CM"
            elif(k==6):
              xtitle = "#pi^{+} cos(#theta) #Delta^{++}-helicity"
            elif(k==7):
              xtitle = "#pi^{+} #phi #Delta^{++}-helicity"
            elif(k==8):
              xtitle = "#pi^{-} cos(#theta) #Delta^{0}-helicity"
            elif(k==9):
              xtitle = "#pi^{-} #phi #Delta^{0}-helicity"
            elif(k==10):
              xtitle = "#pi^{+} cos(#theta) #rho-helicity"
            elif(k==11):
              xtitle = "#pi^{+} #phi #rho-helicity"

            h[f][i][j][k][n].GetXaxis().SetTitle(xtitle)

            h[f][i][j][k][n].SetLineWidth(2)
            if(i==0): 
              h[f][i][j][k][n].SetFillColor(5)
            elif(i==2):
              h[f][i][j][k][n].SetFillColor(0)
              h[f][i][j][k][n].SetLineColor(2)
              h[f][i][j][k][n].SetLineWidth(8)
              nbins = h[f][0][j][k][n].GetNbinsX()
              if float(h[f][i][j][k][n].Integral(1,nbins)) != 0.0:
                scaleby =  float(h[f][0][j][k][n].Integral(1,nbins)) / float(h[f][i][j][k][n].Integral(1,nbins))
                h[f][i][j][k][n].Scale( scaleby )
            elif i==1 or i==3:
              if i==1:
                h[f][i][j][k][n].SetLineWidth(8)
              elif i==3:
                h[f][i][j][k][n].SetLineWidth(2)
                h[f][i][j][k][n].SetMarkerSize(0.2)
              h[f][i][j][k][n].SetMarkerColor(8*n+33)
              h[f][i][j][k][n].SetLineColor(8*n+33)

            # Scale the raw to get cross sections
            if i==3:
              scalefactor = 1.0/h[f][i][j][k][n].GetBinWidth(1)
              h[f][i][j][k][n].Scale(scalefactor)
              h[f][i][j][k][n].SetMinimum(0.0)

            if k>=3: 
              h[f][i][j][k][n].GetXaxis().SetRangeUser(-1.5, 1.5)

            if i==0:
              h[f][i][j][k][n].DrawCopy("h")
              #print "drawing....."
            elif i==2:
              h[f][i][j][k][n].DrawCopy("same")
              #print "drawing same....."

            if n==0 and i==0 and j==rawtop:
              name = "%d-%d" % (wbinlo[f],wbinhi[f])
              tpl_wbin[f].append(TPaveLabel(0.5,0.8,0.99,0.99,name,"NDC"))
              tpl_wbin[f][k].SetFillStyle(1)
              tpl_wbin[f][k].SetFillColor(1)
              tpl_wbin[f][k].SetTextColor(0)
              tpl_wbin[f][k].Draw()

            h[f][i][j][k][n].SetMaximum(1.3*h[f][i][j][k][n].GetMaximum())
            if i==3:
              if k<=2:
                h[f][i][j][k][n].SetMaximum(1100)
              elif k>2 and k<=5:
                h[f][i][j][k][n].SetMaximum(100)
              else:
                h[f][i][j][k][n].SetMaximum(100)




#
# Calculate chi2 for the plots
#
print "Calculate chi2.......\n"
chi2 = []
ndf = []
for f in range(0,numPlots):
  chi2.append([])
  ndf.append([])
  for n in range(0,numiter):
    chi2[f].append(0.0)
    ndf[f].append(0.0)
    #for j in range(1,4): # 1357
    for j in range(0,4): 
      for k in range(0,12):
        nbins = h[f][0][j][k][0].GetNbinsX()
        if  nbins>0:
          for p in range(1,nbins):
            x0 = h[f][0][j][k][0].GetBinContent(p)
            x1 = h[f][1][j][k][n].GetBinContent(p)
            xerr0 = h[f][0][j][k][0].GetBinError(p)
            xerr1 = h[f][1][j][k][n].GetBinError(p)
            if  not (x0 == 0 and x1 == 0):
              chi2[f][n] += pow(x0-x1,2)/(xerr0*xerr0 + xerr1*xerr1)
              ndf[f][n] += 1
    if ndf[f][n] != 0:
      chi2[f][n] /= ndf[f][n]
    else:
      chi2[f][n] = 0.0

    print "\tf/n: " + str(f) + " " + str(n) 
    if chi2[f][n] < bestchi2[f]:
      bestchi2[f] = chi2[f][n]

      #print "\tf/n: " + str(f) + " " + str(n) + " " + str(chi2[f][n]) + " " + str(bestchi2[f]) + " " + str(chi2[f][n]-bestchi2[f])


# 
# Calcluate the systematic errors
#
for f in range(0,numPlots):
  for n in range(0,numiter):
    diff = chi2[f][n] - bestchi2[f]
    if diff==0.0:
      #for i in range(1,4,2): #types
      for k in range(0,15): # kinvars
        xdval = [0.0, 0.0, 0.0, 0.0]
        xaval = [0.0, 0.0, 0.0, 0.0]
        errs = [0.0, 0.0, 0.0, 0.0]
        #for j in range(0,4): # topologies
        nbins = h[f][0][0][k][0].GetNbinsX()
        if  nbins>0:
          for p in range(1,nbins):
            for j in range(0,4): # topologies
              xdval[j] = h[f][0][j][k][0].GetBinContent(p) # data
              xaval[j] = h[f][1][j][k][n].GetBinContent(p) # accwt
              errs[j] = 0.0
              if xdval[j] != 0.0:
                errs[j] = abs(xdval[j]-xaval[j])/xdval[j]

            toterrs = 0.0
            npts = 0
            for j in range(0,4): # topologies
              if errs[j] != 0:
                toterrs += errs[j]*errs[j]
                npts += 1
                print str(j) + " " + str(errs[j])
            if npts!=0:
              toterrs = math.sqrt(toterrs)
              #toterrs /= float(npts)
              print "toterrs/npts: " + str(toterrs) + " " + str(npts)


            for j in range(0,4): # topologies
              cspoint = h[f][3][j][k][n].GetBinContent(p) # rawwt
              midpoint = h[f][3][j][k][n].GetBinCenter(p) # rawwt
              err = cspoint*toterrs
              if cspoint != 0.0:
                print "x2/err: " + str(cspoint) + " " + str(toterrs)
                if toterrs>1.00:
                  cspoint = 0.0
                  err = 0.0
                  #h[f][3][j][k][n].SetBinContent( p, 0.0)
                  #h[f][3][j][k][n].SetBinError( p, 0.0)
                #else:
                  #h[f][3][j][k][n].SetBinError( p, cspoint*toterrs )
                h[f][3][j][k][n].SetBinContent( p, cspoint)
                h[f][3][j][k][n].SetBinError( p, err)
              ############################################################
              ########## XML #############################################
              ############################################################
              output = "<event x=\"" + str(midpoint) + "\" "
              output += "val=\"" + str(cspoint) + "\" "
              output += "err=\"" + str(err) + "\" "
              output += ">\n"
              print output
        

#
# Plot the ones which pass our chi2 cut
#
for f in range(0,numPlots):
  firstOne = [1,1,1, 1,1,1, 1,1,1 ,1,1,1, 1,1,1]
  for n in range(0,numiter):
    diff = chi2[f][n] - bestchi2[f]
    print "chi2/best: " + str(chi2[f][n]) + " " + str(bestchi2[f])
    if  diff <= chi2cut:
      #print "\tChoosing iteration: " + str(n) + " based on " + str(diff) + " is less than " + str(chi2cut)
      output = str(wbinlo[f]) + " " + str(wbinhi[f]) + " " + str(n) + "\n"
      OUT.write(output)
      for i in range(1,4,2): # types of plots data/accwt/acc/rawwt/raw
        for j in range(0,4): # topology
          for k in range(0,15):
            whichcan = 0
            whichpad = 0
            if i<3: 
              whichcan = k/3 + j*4
              whichpad = f + numPlots*(k%3) + numPlots+1
            else: 
              whichcan = k/3 + 16
              whichpad = f + numPlots*(k%3) + numPlots+1

            #print "whichcan/whichpad: " + str(whichcan) + " " + str(whichpad)
            bottom[whichcan].cd(whichpad);
            gPad.SetBottomMargin(0.18);
            gPad.SetLeftMargin(0.18);

            if h[f][i][j][k][n] and k<12:
              if i==1:
                h[f][i][j][k][n].DrawCopy("esame")
                gPad.Update()
              elif i==3 and j==rawtop:
                #print str(i) + "\twhichcan/whichpad: " + str(whichcan) + " " + str(whichpad)
                if firstOne[k]:
                  #if k>=3: 
                    #h[f][i][j][k][n].GetXaxis().SetRangeUser(-1.5, 1.5)
                  h[f][i][j][k][n].DrawCopy("e")
                  gPad.Update()
                  firstOne[k] = 0
                else:
                  if k>=3: 
                    h[f][i][j][k][n].GetXaxis().SetRangeUser(-1.5, 1.5)
                  h[f][i][j][k][n].DrawCopy("esame")
                  gPad.Update()

            # 2D plots
            if h2d[f][i][j][k][n] and (i==3 and j==rawtop):
              bottom[whichcan + 4].cd(whichpad)
              gPad.SetBottomMargin(0.18)
              gPad.SetLeftMargin(0.18)
              gPad.SetRightMargin(0.25)
              gPad.SetLogz()
              ####
              # Scale the 2D plots
              ####
              h2d[f][i][j][k][n].RebinX(4)
              h2d[f][i][j][k][n].RebinY(4)
              xwidth = 1.0/h2d[f][i][j][k][n].GetXaxis().GetBinWidth(1)
              ywidth = 1.0/h2d[f][i][j][k][n].GetYaxis().GetBinWidth(1)
              scalefactor = 1.0/(xwidth*ywidth) # To turn to microbarns
              h2d[f][i][j][k][n].Scale(scalefactor)

              zmax = 1
              if k==0:
                zmax=0.0032
              elif k==1:
                zmax=0.30e-3
              elif k==2:
                zmax=0.30e-3

              elif k==3:
                zmax=0.0018
              elif k==4:
                zmax=0.0018
              elif k==5:
                zmax=0.0015

              elif k==6:
                zmax=0.0020
              elif k==7:
                zmax=0.0020
              elif k==8:
                zmax=0.0015

              elif k==9:
                zmax=0.0024
              elif k==10:
                zmax=0.0015
              elif k==11:
                zmax=0.35e-3

              elif k==12:
                zmax=0.0014
              elif k==13:
                zmax=0.0014
              elif k==14:
                zmax=0.0022

              """
              if k==0:
                zmax=0.010
              elif k==1:
                zmax=0.0008
              elif k==2:
                zmax=0.0008

              elif k==3:
                zmax=0.005
              elif k==4:
                zmax=0.005
              elif k==5:
                zmax=0.005

              elif k==6:
                zmax=0.005
              elif k==7:
                zmax=0.005
              elif k==8:
                zmax=0.004

              elif k==9:
                zmax=0.006
              elif k==10:
                zmax=0.004
              elif k==11:
                zmax=0.0008
              """

              h2d[f][i][j][k][n].SetMaximum(zmax)
              h2d[f][i][j][k][n].SetMinimum(0)
              h2d[f][i][j][k][n].DrawCopy("colz")
              gPad.Update()



#
# Draw some legends
#
leg = []
for f in range(0,numPlots):
  leg.append([])
  for k in range(0,numcan):
    leg[f].append(TLegend(0.01, 0.01, 0.99, 0.99))
    for n in range(0,numiter):
      name = "-log(L): %3.3f #Delta #Chi^{2}: %3.3f" % (logL[f][n]-bestlogL[f], chi2[f][n]-bestchi2[f])
      leg[f][k].AddEntry(h[f][1][0][0][n], name, "l")
      name = "#Chi^{2}/ndf: %3.3f/%3.3f" % (chi2[f][n]*ndf[f][n], ndf[f][n])
      leg[f][k].AddEntry(h[f][1][0][0][n], name, "")

    bottom[k].cd(f+1)
    leg[f][k].Draw()
    gPad.Update()


#
# Draw the plots for the talk
#
isobarlines = [ [] , [] ]
linecount = 0
cantalk = []
toptalk = []
bottalk = []
hwline = []
for f in range(0,numPlots):
  cantalk.append([])
  toptalk.append([])
  bottalk.append([])
  hwline.append([])
  for num in range(0,6):
    name = "cantalk%d" % (5*f + num)
    if num==0:
      cantalk[f].append(TCanvas(name,name,100+10*f + 150*num, 100+10*f, 950, 600))

    else:
      cantalk[f].append(TCanvas(name,name,100+10*f + 150*num, 100+10*f, 900, 300))

    cantalk[f][num].SetFillColor(0)

    name = "toptalk%d" % (5*f + num)
    toptalk[f].append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
    toptalk[f][num].SetFillColor(0)
    toptalk[f][num].Draw()
    name = "bottalk%d" % (5*f + num)
    bottalk[f].append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.85))
    bottalk[f][num].SetFillColor(0)
    bottalk[f][num].Draw()

    if num==0:
      bottalk[f][num].Divide(4, 3)
    else:
      bottalk[f][num].Divide(3,1)

    name = "hwline%d_%d" % (f, num)
    hwline[f].append(TH1F(name, name, 81, 1395, 2205))
    for p in range(1400,wbinhi[f],10):
      print str(p) + " " + str(f) + " " + str(num)
      hwline[f][num].Fill(p)
    hwline[f][num].SetFillColor(2)
    hwline[f][num].SetTitle("")
    hwline[f][num].GetYaxis().SetNdivisions(0)

    hwline[f][num].GetXaxis().SetNdivisions(6)
    hwline[f][num].GetXaxis().SetLabelSize(0.30)
    hwline[f][num].GetXaxis().SetTitleSize(0.20)
    hwline[f][num].GetXaxis().SetTitleFont(42)
    hwline[f][num].GetXaxis().SetTitleOffset(1.1)
    #hwline[f][num].GetXaxis().SetTitle("W GeV/c^{2}")
    hwline[f][num].GetXaxis().CenterTitle()
    hwline[f][num].SetMaximum(0.5)

    toptalk[f][num].cd()
    gPad.SetBottomMargin(0.28)
    hwline[f][num].Draw()




  #### 
  # Plot the diff cs
  ####
  #for i in range(0,5): # types of plots data/accwt/acc/rawwt/raw
  i=3
  firstOne = [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]
  for n in range(0,5): # iterations
    diff = chi2[f][n] - bestchi2[f]
    print "chi2/best: " + str(chi2[f][n]) + " " + str(bestchi2[f])
    if  diff <= chi2cut:
      #for j in range(0,4): # topologies
      if(1):
        j=rawtop
        for k in range(0,15): # kinvars
          whichcan = 0
          whichpad = 1

          if k==0:
            whichpad=1
          elif k==1:
            whichpad=5
          elif k==2:
            whichpad=9

          elif k==3:
            whichpad=2
          elif k==4:
            whichpad=6
          elif k==5:
            whichpad=10

          elif k==6:
            whichpad=3
          elif k==7:
            whichpad=7
          elif k==8:
            whichpad=11

          elif k==9:
            whichpad=4
          elif k==10:
            whichpad=8
          elif k==11:
            whichpad=12

          #elif k==12:
            #whichpad=3
          #elif k==13:
            #whichpad=8
          #elif k==14:
            #whichpad=13

          bottalk[f][whichcan].cd(whichpad)
          gPad.SetBottomMargin(0.18)
          gPad.SetLeftMargin(0.18)
          if firstOne[k] and k<12:
            #if k>=12:
              #gPad.SetLogy()
              #h[f][i][j][k][n].Rebin(4)
              #h[f][i][j][k][n].SetMaximum(30.0)
              #h[f][i][j][k][n].SetMinimum(0.1)
              #h[f][i][j][k][n].GetXaxis().SetRangeUser(-0.5, 1.5)
            #if j==rawtop and k==0:
            print str(f) + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(n) + "\n"
            h[f][i][j][k][n].SetLineColor(2)
            h[f][i][j][k][n].SetMarkerColor(2)
            h[f][i][j][k][n].DrawCopy("e")
            tpl_wbin[f][0].Draw()
            gPad.Update()
            firstOne[k] = 0
          elif k<12:
            if k>=3: 
              h[f][i][j][k][n].GetXaxis().SetRangeUser(-1.0, 1.0)
            h[f][i][j][k][n].SetLineColor(2)
            h[f][i][j][k][n].SetMarkerColor(2)
            h[f][i][j][k][n].DrawCopy("esame")
            tpl_wbin[f][0].Draw()
            gPad.Update()

          # 2D plots
          if h2d[f][i][j][k][n] and (i==3 and j==rawtop):
            whichcan = 0
            whichpad = 1
            if k==0:
              whichcan=1
              whichpad=1
            elif k==1:
              whichcan=1
              whichpad=2
            elif k==2:
              whichcan=1
              whichpad=3
            elif k==3:
              whichcan=2
              whichpad=1
            elif k==4:
              whichcan=2
              whichpad=2
            elif k==5:
              whichcan=2
              whichpad=3
            elif k==6:
              whichcan=3
              whichpad=1
            elif k==7:
              whichcan=3
              whichpad=2
            elif k==8:
              whichcan=3
              whichpad=3
            elif k==9:
              whichcan=4
              whichpad=1
            elif k==10:
              whichcan=4
              whichpad=2
            elif k==11:
              whichcan=4
              whichpad=3

            elif k==12:
              whichcan=5
              whichpad=1
            elif k==13:
              whichcan=5
              whichpad=2
            elif k==14:
              whichcan=5
              whichpad=3

            bottalk[f][whichcan].cd(whichpad)
            gPad.SetBottomMargin(0.18)
            gPad.SetLeftMargin(0.18)
            gPad.SetRightMargin(0.25)
            if doLogz:
              gPad.SetLogz()
            h2d[f][i][j][k][n].DrawCopy("colz")
            tpl_wbin[f][0].Draw()
            # Draw the isobar lines
            lox = h2d[f][i][j][k][n].GetXaxis().GetXmin()
            hix = h2d[f][i][j][k][n].GetXaxis().GetXmax()
            loy = h2d[f][i][j][k][n].GetYaxis().GetXmin()
            hiy = h2d[f][i][j][k][n].GetYaxis().GetXmax()
            delta = 1.232*1.232
            rho = 0.770*0.770
            if k==0:
              isobarlines[0].append(TLine(lox, delta, hix, delta))
              isobarlines[1].append(TLine(delta, loy, delta, hiy))
            elif k==1 or k==2:
              isobarlines[0].append(TLine(lox, delta, hix, delta))
              isobarlines[1].append(TLine(rho, loy, rho, hiy))

            if k<3:
              for l in range(0,2):
                isobarlines[l][linecount].SetLineStyle(2)
                isobarlines[l][linecount].SetLineWidth(4)
                isobarlines[l][linecount].Draw()
              linecount += 1
            gPad.Update()

  for num in range(0,6):
    if num==0:
      name = "plots/can_1Dplots_%s_%d.eps" % (tag, f+offset)
      print name
      cantalk[f][num].SaveAs(name)

    else:
      name = "plots/can_2Dplots_%d_%s_%d.eps" % (num, tag, f+offset)
      print name
      cantalk[f][num].SaveAs(name)


#
# Make some single plots for us
#
if doSingles:
  cansingle = []
  for f in range(0,numPlots):
    cansingle.append([])
    for c in range(0,300):
      name = "can%d_%d" % (f, c)
      cansingle[f].append(TCanvas(name, name, 10+10*(c%48) + 50*(c/48), 10+10*(c%48), 200, 200))
      cansingle[f][c].SetFillColor(0)
      cansingle[f][c].Divide(0)

      cansingle[f][c].cd(1)
      gPad.SetBottomMargin(0.18)
      gPad.SetLeftMargin(0.18)
      gPad.SetRightMargin(0.25)

      name = "singleplots/"
      for n in range(0,5): # iterations
        diff = chi2[f][n] - bestchi2[f]
        if  diff == 0.0:
          if c<48:
            j = c/12
            k = c%12
            h[f][0][j][k][0].Draw("h") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_1Ddata_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=48 and c<96:
            j = (c-48)/12
            k = c%12
            #print str(f) + " " + str(j) + " " + str(k)
            h[f][0][j][k][0].Draw("h") # file/type/topology/kinvar/iteration
            h[f][2][j][k][0].Draw("hsame") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_1Dacc_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=96 and c<144:
            j = (c-96)/12
            k = c%12
            h[f][0][j][k][0].Draw("h") # file/type/topology/kinvar/iteration
            h[f][1][j][k][n].SetLineColor(4)
            h[f][1][j][k][n].Draw("hsame") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_1Daccwt_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=144 and c<192:
            j = (c-144)/12
            k = c%12
            h2d[f][0][j][k][0].Draw("colz") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_2Ddata_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=192 and c<240:
            j = (c-192)/12
            k = c%12
            h2d[f][1][j][k][0].Draw("colz") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_2Daccwt_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=240 and c<288:
            j = (c-240)/12
            k = c%12
            h2d[f][2][j][k][0].Draw("colz") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_2Dacc_%d_%d.eps" % (wbinlo[f], j, k)
          elif c>=288 and c<300:
            j = (c-288)/12
            k = c%12
            print str(f) + " " + str(c) + " " + str(j) + " " + str(k) 
            h2d[f][4][j][k][0].Draw("colz") # file/type/topology/kinvar/iteration
            tpl_wbin[f][k].Draw()
            name += "cansingle%d_2Draw_%d_%d.eps" % (wbinlo[f], j, k)


      gPad.Update()
      cansingle[f][c].SaveAs(name)


"""
  ///////////////////////////////////////////////////////////////////////////////

  /*


     for(i=0;i<numcan;i++)
     {
     sprintf(name,"plots/dsigma_%s_%s_%d_%d.eps",tag,fit,lo,i);
     can[i].SaveAs(name);
     }

     for(int i=0;i<10;i++)
     {
     for(int j=0;j<numiter;j++)
     {
     for(int k=0;k<12;k++)
     {
  //delete inFile[i][j][k];
  }
  }
  }
   */

}
"""
## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]

for f in range(0,numPlots):
  inFile[f].Close()
