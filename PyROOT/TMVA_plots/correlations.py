#!/usr/bin/env python
                                                                                                                                        
                                                                                                                                        # import some modules
import sys
import ROOT
from ROOT import *
from optparse import OptionParser

from color_palette import *

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

batchMode = False

fin = sys.argv[1]
pct_cutoff = float(sys.argv[2])
tag = sys.argv[3]

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



file = TFile(fin)

hName = [ "CorrelationMatrixS", "CorrelationMatrixB" ]
text = []
width = 600
vars = []
vars.append(0)
   
c = TCanvas( "can", "can", 10, 10, 1200, 900)
c.SetFillColor(0)
c.Divide(1,2)

h = []
hnew = []
for ic in range(0,2):
  hdum = file.Get( hName[ic] )
  h.append(hdum)
  hnew.append(hdum.Clone())
  if not hdum:
     print "Did not find histogram " + hName[ic] + " in " + fin

  newMargin1 = 0.13
  newMargin2 = 0.15

  c.cd(ic+1)
  gPad.SetGrid()
  gPad.SetLeftMargin  ( newMargin2 )
  gPad.SetBottomMargin( newMargin2 )
  gPad.SetRightMargin ( newMargin1 )
  gPad.SetTopMargin   ( newMargin1 )

  gStyle.SetHistMinimumZero(0)
  gStyle.SetPaintTextFormat( "3g" )

  h[ic].SetMarkerSize( 1.5 )
  h[ic].SetMarkerColor( 0 ) # Text color
  labelSize = 0.040
  h[ic].GetXaxis().SetLabelSize( labelSize )
  h[ic].GetYaxis().SetLabelSize( labelSize )
  h[ic].LabelsOption( "d" )
  h[ic].SetLabelOffset( 0.011 ) # label offset on x axis    

  h[ic].Draw("colz") # color pads
  c.Update()

  # modify properties of paletteAxis
  paletteAxis = h[ic].GetListOfFunctions().FindObject( "palette" )
  paletteAxis.SetLabelSize( 0.03 )
  paletteAxis.SetX1NDC( paletteAxis.GetX1NDC() + 0.02 )

  h[ic].Draw("textsame") # add text

  # add comment    
  text.append(TText( 0.53, 0.88, "Linear correlation coefficients in %" ))
  text[ic].SetNDC()
  text[ic].SetTextSize( 0.026 )
  text[ic].AppendPad()

  #TMVAGlob.plot_logo( )
  c.Update()
  name = "Plots/correlation_matrix_full_" + tag + ".eps"
  c.SaveAs(name)

  fname = "plots/"
  fname += hName[ic]
  #TMVAGlob::imgconv( c, fname );

##############################################
# Figure out who to remove
##############################################
num_corrs = []
var_corrs = []
good_vars = []
good_vars.append(1) # Fill the first one with a placeholder
num_corrs.append(0) # Fill the first one with a placeholder
var_corrs.append([]) # Fill the first one with a placeholder
firstTime = 1

#######
# Grab the info and print first time
#######
for ic in range(0,2):
  print " ----------- "
  print hName[ic]
  for j in range(1,h[ic].GetNbinsX()+1):
    output = "%12s: " % h[ic].GetXaxis().GetBinLabel(j) 
    if ic==0:
      good_vars.append(1)
      num_corrs.append(0)
      var_corrs.append([])
      vars.append(h[ic].GetXaxis().GetBinLabel(j) )
    for k in range(1,h[ic].GetNbinsY()+1):
      if abs(h[ic].GetBinContent(j,k)) > pct_cutoff and j!=k and not (h[ic].GetYaxis().GetBinLabel(k) in var_corrs[j]):
        num_corrs[j] += 1
        var_corrs[j].append(h[ic].GetYaxis().GetBinLabel(k))
      entry = "%5d & " % h[ic].GetBinContent(j,k)
      output += entry
    print output

######
# Print out the first numbers
######
ic = 0
max_vars = h[ic].GetNbinsX()
not_finished = 1
removed_vars = []
while not_finished == 1:
  max_corrs = 0
  print " ----------- "
  for j in range(1,max_vars+1):
    if good_vars[j] > 0:
      output = "%12s: %d" % (vars[j] , num_corrs[j])
    else:
      output = "\033[42m%12s\033[0m: %d" % (vars[j] , num_corrs[j])

    for k in range(0,len(var_corrs[j])):
      entry = "%12s " % var_corrs[j][k]
      output += entry
    print output

    # Check to see if this is the biggest
    if num_corrs[j] > max_corrs:
      max_corrs = num_corrs[j]

  print "max_corrs: " + str(max_corrs)
  for j in range(1,max_vars+1):
    if num_corrs[j] == max_corrs and not (vars[j] in removed_vars) and good_vars[j] == 1:
      # Add to list of removed vars and mark it as bad
      removed_vars.append(vars[j])
      good_vars[j] = 0
      num_corrs[j] = 0
      # Remove this from other correlations
      for i in range(1,max_vars+1):
        for k in range(0, len(var_corrs[i])):
          if (vars[j] in var_corrs[i]):
            var_corrs[i].remove(vars[j])
            if j!=i and not (vars[i] in removed_vars):
              num_corrs[i] -= 1

  # Check to see if we are finished
  not_finished = 0
  for j in range(1,max_vars+1):
    #print num_corrs[j]
    if num_corrs[j] > 0:
      not_finished = 1

   
############################
# Print the final vals
############################
print " ----------- "
print " ---FINAL--- "
print " ----------- "
for j in range(1,max_vars+1):
  if good_vars[j] > 0:
    output = "%12s: %d" % (vars[j] , num_corrs[j])
  else:
    output = "\033[42m%12s\033[0m: %d" % (vars[j] , num_corrs[j])

  #for k in range(0,len(var_corrs[j])):
    #entry = "%12s " % var_corrs[j][k]
    #output += entry
  print output


cafter = TCanvas( "canafter", "canafter", 20, 20, 1200, 900)
cafter.SetFillColor(0)
cafter.Divide(1,2)

box = []
boxcount = 0
for ic in range(0,2):
  cafter.cd(ic+1)
  gPad.SetGrid()
  gPad.SetLeftMargin  ( newMargin2 )
  gPad.SetBottomMargin( newMargin2 )
  gPad.SetRightMargin ( newMargin1 )
  gPad.SetTopMargin   ( newMargin1 )

  gStyle.SetPaintTextFormat( "3g" )

  for j in range(1,max_vars+1):
    if good_vars[j] == 0:
      # Horizontal
      xlo = hnew[ic].GetBinLowEdge(1)
      xhi = hnew[ic].GetBinLowEdge(max_vars+1)
      ylo = hnew[ic].GetBinLowEdge(j)
      yhi = hnew[ic].GetBinLowEdge(j+1)
      box.append(TBox(xlo, ylo, xhi, yhi))
      box[boxcount].SetFillColor(1)
      boxcount +=1 
      # Verstical
      xlo = hnew[ic].GetBinLowEdge(j)
      xhi = hnew[ic].GetBinLowEdge(j+1)
      ylo = hnew[ic].GetBinLowEdge(1)
      yhi = hnew[ic].GetBinLowEdge(max_vars+1)
      box.append(TBox(xlo, ylo, xhi, yhi))
      box[boxcount].SetFillColor(1)
      boxcount +=1 

  #hnew[ic].Draw()
  hnew[ic].SetMarkerSize( 1.5 )
  hnew[ic].SetMarkerColor( 1 ) # Text color
  labelSize = 0.040
  hnew[ic].GetXaxis().SetLabelSize( labelSize )
  hnew[ic].GetYaxis().SetLabelSize( labelSize )
  hnew[ic].LabelsOption( "d" )
  hnew[ic].SetLabelOffset( 0.011 ) # label offset on x axis    

  hnew[ic].Draw("colz") # color pads
  cafter.Update()

  # modify properties of paletteAxis
  paletteAxis = hnew[ic].GetListOfFunctions().FindObject( "palette" )
  paletteAxis.SetLabelSize( 0.03 )
  paletteAxis.SetX1NDC( paletteAxis.GetX1NDC() + 0.02 )

  hnew[ic].Draw("textsame") # add text
  for b in range(0, boxcount):
    box[b].Draw()

  # add comment    
  text.append(TText( 0.53, 0.88, "Linear correlation coefficients in %" ))
  text[ic+2].SetNDC()
  text[ic+2].SetTextSize( 0.026 )
  text[ic+2].AppendPad()

  #TMVAGlob.plot_logo( )
  cafter.Update()

##########################################################################
# Plot portions of the correlation matrix
##########################################################################
cportion = []
gStyle.SetOptStat(0)
for i in range(0,2):
  name = "cportion" + str(i)
  cportion.append(TCanvas( name, name, 50+i*10, 50+i*10, 1200, 200))
  cportion[i].SetFillColor(0)
  cportion[i].Divide(1,1)

hportion = []
for ic in range(0,2):
  cportion[ic].cd(1)
  newMargin1 = 0.13
  newMargin2 = 0.15

  gPad.SetGrid()
  gPad.SetLeftMargin  ( 0.10 )
  gPad.SetBottomMargin( 0.40 )
  gPad.SetRightMargin ( newMargin1 )
  gPad.SetTopMargin   ( newMargin1 )

  nbins = hnew[ic].GetNbinsX()
  name = "hportion" + str(ic)
  hdum = TH2F(name, name, nbins, 1, nbins+1, 2, 1, 3)
  for j in range(1,3):
    yname = hnew[ic].GetYaxis().GetBinLabel(j)
    for i in range(1, nbins+1):
      xname = hnew[ic].GetXaxis().GetBinLabel(i)
      hdum.Fill(xname, yname, hnew[ic].GetBinContent(i, j))

  hportion.append(hdum)

  gStyle.SetPaintTextFormat( "3g" )

  hportion[ic].SetTitle("")

  hportion[ic].SetMarkerSize( 3.5 )
  hportion[ic].SetMarkerColor( 1 ) # Text color
  labelSize = 0.040
  hportion[ic].GetXaxis().SetLabelOffset( 1.6 )
  hportion[ic].GetXaxis().SetLabelSize( 0.11 )

  hportion[ic].GetYaxis().SetLabelSize( 0.12 )

  hportion[ic].SetMaximum( 100)
  hportion[ic].SetMinimum(-100)

  hportion[ic].LabelsOption( "d" )
  #hportion[ic].LabelsOption( "v" )
  hportion[ic].SetLabelOffset( 0.04 ) # label offset on x axis    

  hportion[ic].Draw("colz") # color pads
  cportion[ic].Update()
  # modify properties of paletteAxis
  paletteAxis = hportion[ic].GetListOfFunctions().FindObject( "palette" )
  paletteAxis.SetLabelSize( 0.03 )
  paletteAxis.SetX1NDC( paletteAxis.GetX1NDC() + 0.02 )

  hportion[ic].Draw("textsame") # add text
  cportion[ic].Update()

  name = "Plots/correlation_matrix_portion_" + str(ic) + "_" + tag + ".eps"
  cportion[ic].SaveAs(name)


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
                                                                                                                                                

