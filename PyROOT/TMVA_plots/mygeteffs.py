#!/usr/bin/env python
                                                                                                                                        
                                                                                                                                        # import some modules
import sys
import ROOT
from ROOT import *
from optparse import OptionParser

from color_palette import *

from array import *

####################################################################
# this macro plots the correlation matrix of the various input
# variables used in TMVA (e.g. running TMVAnalysis.C).  Signal and
# Background are plotted separately
#
# input: - Input file (result from TMVA),
#        - use of colors or grey scale
#        - use of TMVA plotting TStyle
####################################################################
#void correlations( TString fin = "TMVA.root", Bool_t greyScale = kFALSE, Bool_t useTMVAStyle = kTRUE )

##################
# Use cool palette
##################
set_palette()

gStyle.SetFrameFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.20)

batchMode = False
firstPlotToAverage = 0

###############################################
# Parse the arguments
###############################################
parser = OptionParser()
parser.add_option("-t", "--tag", dest="tag", default = "default", help="Tag for output files")
parser.add_option("-b", "--best-iter", dest="best_iter", default = -1, help="Best iteration to match")
parser.add_option("-n", "--nbkg", dest="nbkg", default = 10, help="Number of background events for Punzi")
parser.add_option("-a", "--asig", dest="asig", default = 0, help="a/Sigma significance for Punzi")
parser.add_option("--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("--plots-for-talk", dest="plots_for_talk", action = "store_true", default = False, help="Produce the plots for a talk on this material")

(options, args) = parser.parse_args()

#my_best_iter = -1
my_best_iter = int(options.best_iter)
nbkg = float(options.nbkg)
asig = int(options.asig)

# Input files
fin = args

###############################################



methods = [ 'BDT', 'MLP', 'RuleFit', 'Likelihood', 'Fisher' ]
hNameTags = [ "_S", "_B", "_rejBvsS" ]

nmethods = len(methods)
nhistos = len(hNameTags)
nfiles = len(fin)
   
rowcol = int(sqrt(len(methods))) + 1

can = []
for f in range(0, len(fin) + 2):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1200, 900))
  can[f].SetFillColor(0)
  if f< len(fin):
    can[f].Divide(nmethods,nhistos+1)
  else:
    can[f].Divide(rowcol, rowcol)

########################################################

mygr = []
h = []
rootfile = []
xpts = []
ypts = []
for f in range(0, len(fin)):
  rootfile.append(TFile(fin[f]))
  h.append([])
  mygr.append([])
  xpts.append([])
  ypts.append([])
  for i in range(0,nmethods):
    h[f].append([])
    rootfile[f].cd("Method_"+methods[i]+"/"+methods[i])
    #gDirectory.ls()
    for j in range(0,nhistos):
      name = "MVA_" + methods[i] + hNameTags[j]
      #print "looking for " + name
      hdum = gROOT.FindObject( name )
      if (hdum):
        #print "found: " + name
        h[f][i].append(hdum)

        can[f].cd(j*nmethods + i + 1)

        h[f][i][j].SetMarkerSize( 1.5 )
        h[f][i][j].SetMarkerColor( 1 ) # Text color
        h[f][i][j].SetLabelOffset( 0.011 ) # label offset on x axis    

        h[f][i][j].SetLineWidth(3)
        h[f][i][j].SetLineColor(i + 1)

        h[f][i][j].Draw("c") # color pads
        can[f].Update()

  for i in range(0,nmethods):
    xpts[f].append(array('f'))
    ypts[f].append(array('f'))

    nbins = h[f][i][0].GetNbinsX()
    totx = h[f][i][0].Integral()
    toty = h[f][i][1].Integral()
    for p in range(1,nbins+1):
      xpts[f][i].append(h[f][i][0].Integral(p,nbins)/totx)
      ypts[f][i].append(1.0 - h[f][i][1].Integral(p,nbins)/toty)

    mygr[f].append(TGraph(len(xpts[f][i]), xpts[f][i], ypts[f][i]))

    can[f].cd(2*nmethods + i + 1)
    
    mygr[f][i].SetTitle()

    mygr[f][i].GetYaxis().CenterTitle()
    mygr[f][i].GetYaxis().SetNdivisions(4)
    mygr[f][i].GetYaxis().SetLabelSize(0.06)
    mygr[f][i].GetYaxis().SetTitleSize(0.09)
    mygr[f][i].GetYaxis().SetTitleOffset(0.9)
    mygr[f][i].GetYaxis().SetTitle("Bkg rej (1-#eta_{B})")

    mygr[f][i].GetXaxis().SetNdivisions(6)
    mygr[f][i].GetXaxis().CenterTitle()
    mygr[f][i].GetXaxis().SetLabelSize(0.06)
    mygr[f][i].GetXaxis().SetTitleSize(0.09)
    mygr[f][i].GetXaxis().SetTitleOffset(1.0)
    mygr[f][i].GetXaxis().SetTitle("Signal eff (#epsilon_{S})")

    mygr[f][i].GetYaxis().SetRangeUser(0.2, 1.0)
    mygr[f][i].GetXaxis().SetRangeUser(0.2, 1.0)


    mygr[f][i].SetMarkerStyle(20)
    mygr[f][i].SetLineColor(i + 1)
    mygr[f][i].SetMarkerSize(0.8)
    if f<10:
      mygr[f][i].SetMarkerColor(2)
    elif f>=10 and f<15:
      mygr[f][i].SetMarkerColor(4)
    elif f>=15 and f<20:
      mygr[f][i].SetMarkerColor(8)

    # For all the same green
    #mygr[f][i].SetMarkerColor(8)
    # Special for the best one
    if f==my_best_iter:
      mygr[f][i].SetMarkerColor(7)
      mygr[f][i].SetMarkerSize(1.5)

    # For all the same green
    mygr[f][i].Draw("p")
    can[f].Update()
   
###################################################
# Plot the graphs
###################################################
for f in range(0, len(fin)):
  for j in range(0,nmethods):
    can[len(fin)].cd(j+1)
    if f==0:
      mygr[f][j].Draw("ap")
    else:
      mygr[f][j].Draw("p")

    can[len(fin)].Update()

###################################################
#
###################################################
xnewpts = []
ynewpts = []
grnew = []
grmean = []
xmeanpts = []
ymeanpts = []
for j in range(0,nmethods):
  grnew.append([])
  xnewpts.append([])
  ynewpts.append([])
  xmeanpts.append(array('f'))
  ymeanpts.append(array('f'))
  for f in range(0, len(fin)):
    xnewpts[j].append(array('f'))
    ynewpts[j].append(array('f'))

    can[len(fin)+1].cd(j+1)
    ndiv = 0.01
    npts = len(ypts[f][j])
    for p in range(2,98):
      pt = p * ndiv
      min_above_diff = 100000.0
      min_below_diff = 100000.0
      closest_above = 0
      closest_below = 0
      for n in range(0,npts):
        diff_above = ypts[f][j][n] - pt
        diff_below = ypts[f][j][n] - pt
        #print str(pt) + " " + str(diff_above) + " " + str(diff_below)+ " " + str(min_above_diff) + " " + str(min_below_diff)

        if min_above_diff > fabs(diff_above) and diff_above > 0:
          min_above_diff = diff_above
          closest_above = n

        #print "here" +  str(pt) + " " + str(diff_below)+ " " + str(min_below_diff)
        if fabs(min_below_diff) > fabs(diff_below) and diff_below < 0:
          min_below_diff = diff_below
          closest_below = n

      #print str(f) + " " + str(j) + " " + str(closest_below) + " " + str(closest_above)
      y0 = ypts[f][j][closest_below]
      y1 = ypts[f][j][closest_above]
      x0 = xpts[f][j][closest_below]
      x1 = xpts[f][j][closest_above]

      m = (x1-x0)/(y1-y0)

      ynew = pt
      xnew = x0 + m*(pt-y0)
      xnewpts[j][f].append(xnew)
      ynewpts[j][f].append(ynew)

      if f==firstPlotToAverage:
        xmeanpts[j].append(xnew)
        ymeanpts[j].append(ynew)
      elif f>firstPlotToAverage:
        xmeanpts[j][p-2] += xnew

    #############################################
    # Build the graphs
    #############################################
    can[nfiles+1].cd(j+1)
    grnew[j].append(TGraph(len(xnewpts[j][f]), xnewpts[j][f], ynewpts[j][f]))

    grnew[j][f].SetTitle()

    grnew[j][f].GetYaxis().CenterTitle()
    grnew[j][f].GetYaxis().SetNdivisions(4)
    grnew[j][f].GetYaxis().SetLabelSize(0.06)
    grnew[j][f].GetYaxis().SetTitleSize(0.09)
    grnew[j][f].GetYaxis().SetTitleOffset(0.9)
    grnew[j][f].GetYaxis().SetTitle("Bkg rej (1-#eta_{B})")

    grnew[j][f].GetXaxis().SetNdivisions(6)
    grnew[j][f].GetXaxis().CenterTitle()
    grnew[j][f].GetXaxis().SetLabelSize(0.06)
    grnew[j][f].GetXaxis().SetTitleSize(0.09)
    grnew[j][f].GetXaxis().SetTitleOffset(1.0)
    grnew[j][f].GetXaxis().SetTitle("Signal eff (#epsilon_{S})")

    grnew[j][f].GetYaxis().SetRangeUser(0.2, 1.0)
    grnew[j][f].GetXaxis().SetRangeUser(0.2, 1.0)

    grnew[j][f].SetMarkerSize(0.8)
    if f<10:
      grnew[j][f].SetMarkerColor(2)
    elif f>=10 and f<15:
      grnew[j][f].SetMarkerColor(4)
    elif f>=15 and f<20:
      grnew[j][f].SetMarkerColor(8)

    # For all the same green
    #grnew[j][f].SetMarkerColor(8)
    # Special for the best one
    if f==my_best_iter:
      grnew[j][f].SetMarkerColor(7)
      grnew[j][f].SetMarkerSize(1.5)

    grnew[j][f].SetMarkerStyle(20)
    if f==0:
      grnew[j][f].Draw("ap")
    else:
      grnew[j][f].Draw("p")
      grnew[j][0].Draw("p")

  ################################
  # Do the means
  ################################
  for n in range(0,len(xmeanpts[j])):
    #print "before: %f %f" % (xmeanpts[j][n], ymeanpts[j][n])
    xmeanpts[j][n] /= float(nfiles - firstPlotToAverage)
    #print "after: %f %f" % (xmeanpts[j][n], ymeanpts[j][n])


  can[nfiles+1].cd(j+1)
  grmean.append(TGraph(len(xmeanpts[j]), xmeanpts[j], ymeanpts[j]))
  grmean[j].SetLineWidth(2)
  grmean[j].SetLineColor(2)
  grmean[j].Draw("l")

  can[len(fin)+1].Update()

#########################################
# Find the most significant point
#########################################
#nbkg = 14.0
#asig = 5.0

line = [[], []]
best_fom = []
best_sigeff = []
best_bkgeff = []
for j in range(0,nmethods):
  print methods[j]
  best_fom.append(0.0)
  best_sigeff.append(0.0)
  best_bkgeff.append(0.0)
  for p in range(0, len(xmeanpts[j])):
    x = xmeanpts[j][p]
    y = ymeanpts[j][p]
    fom = x / (sqrt(nbkg*(1.0 - y)) +  asig/2.0);
    if best_fom[j] < fom:
      best_fom[j] = fom
      best_sigeff[j] = x
      best_bkgeff[j] = 1.0 - y

  print "\tf.o.m: %f  eff_sig: %f    eff_bkg:%f" % (best_fom[j], best_sigeff[j], best_bkgeff[j])
  ###############################
  # Find closest point and file iteration
  ###############################
  best_tot_dist = 1000000.0
  best_tot_iter = -1
  best_dist = 1000000.0
  best_iter = -1
  for f in range(0, nfiles):
    tot_dist = 0
    for p in range(0, len(xnewpts[j][f])):
      x = xnewpts[j][f][p]
      y = ynewpts[j][f][p]
      dist = (best_sigeff[j]-x)**2 + (best_bkgeff[j]-(1.0-y))**2
      tot_dist += dist
      #if j==1:
        #print str(x) + " " + str(y) + " " + str(dist)
      if best_dist > dist:
        best_dist = dist
        best_iter = f
        #if j==1:
          #print "best: " + str(f) + " " + str(x) + " " + str(y) + " " + str(dist) 

    if best_tot_dist > tot_dist:
      best_tot_dist = tot_dist
      best_tot_iter = f
      #if j==1:
        #print "best: " + str(f) + " " + str(x) + " " + str(y) + " " + str(dist) 

  print "\tbest iter: %d" % (best_iter)
  print "\tbest dist: %f" % (best_dist)
  #print "\tbest tot_iter: %d" % (best_tot_iter)
  #print "\tbest tot_dist: %f" % (best_tot_dist)


  can[nfiles+1].cd(j+1)
  line[0].append(TLine(0.2, 1.0-best_bkgeff[j], 1.0, 1.0-best_bkgeff[j]))
  line[1].append(TLine(best_sigeff[j], 0.2, best_sigeff[j], 1.0))
  for i in range(0,2):
    line[i][j].SetLineStyle(2)
    line[i][j].Draw()

  can[nfiles+1].Update()


if options.plots_for_talk:
  ##################################################
  # Build plots for talk
  ##################################################
  classifier = 1

  lorangex = best_sigeff[classifier] - 0.07
  hirangex = best_sigeff[classifier] + 0.07
  lorangey = 1.0-best_bkgeff[classifier] - 0.07
  hirangey = 1.0-best_bkgeff[classifier] + 0.07
  print "ranges: %f %f %f %f\n" % (lorangex, hirangex, lorangey, hirangey) 

  cantalk = []
  for n in range(0, 8):
    cantalk.append([])
    for f in range(0, nfiles + 1):
      name = "cantalk" + str(n) + "_" + str(f)
      cantalk[n].append(TCanvas( name, name, 100+10*f + 20*n, 10+10*f, 700, 500))
      cantalk[n][f].SetFillColor(0)
      cantalk[n][f].Divide(1, 1)

      for i in range(0, f+1):
        cantalk[n][f].cd(1)
        option = "p"
        if i==0:
          option = "ap"

        if i>=nfiles:
          i = nfiles - 1

        if n==0:
          mygr[i][classifier].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
        elif n==1:
          grnew[classifier][i].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
        elif n==2:
          mygr[i][classifier].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
          for k in range(0,2):
            line[k][classifier].Draw()
        elif n==3:
          grnew[classifier][i].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
          for k in range(0,2):
            line[k][classifier].Draw()
        elif n==4:
          mygr[i][classifier].GetXaxis().SetRangeUser(lorangex, hirangex)
          mygr[i][classifier].GetYaxis().SetRangeUser(lorangey, hirangey)
          mygr[i][classifier].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
        elif n==5:
          grnew[classifier][i].GetXaxis().SetRangeUser(lorangex, hirangex)
          grnew[classifier][i].GetYaxis().SetRangeUser(lorangey, hirangey)
          grnew[classifier][i].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
        elif n==6:
          mygr[i][classifier].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
          for k in range(0,2):
            line[k][classifier].Draw()
        elif n==7:
          grnew[classifier][i].Draw(option)
          if f==nfiles:
            grmean[classifier].Draw("l")
          for k in range(0,2):
            line[k][classifier].Draw()

        if best_iter!=-1 and f%2==0:
          if n<4:
            mygr[best_iter][classifier].Draw("p")
          else:
            grnew[classifier][best_iter].Draw("p")

        gPad.Update()

      name = "Plots/sigeff_decisions_%s_%d_%d.eps" % (options.tag, n, f)
      cantalk[n][f].SaveAs(name)

############################################################################
############################################################################


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
                                                                                                                                                

