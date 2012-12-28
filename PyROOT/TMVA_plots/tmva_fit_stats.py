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

filenames = []
for f in range(1,len(sys.argv)):
  filenames.append(sys.argv[f])

nfiles = len(filenames)

########################
# Train (s/b)  Test (s/b)
bs_tt = [[], [], [], []]
######################################
# Loop over the files
######################################
files = []
hnevents = []
for f in range(0, nfiles):
  print filenames[f]
  files.append(open(filenames[f]))
  name = "hnevents" + str(f)
  hnevents.append([])
  for i in range(0, 4):
    hnevents[f].append(TH1F(name, name, 4, 0.5, 3.5))
  for line in files[f]:
    words = line.split()
    if "entries" in words and ("Training" in words or "Testing" in words):
      print words[-1]
      if "Training" in words and "signal" in words:
        bs_tt[0].append(words[-1])
      elif "Training" in words and "background" in words:
        bs_tt[1].append(words[-1])
      elif "Testing" in words and "signal" in words:
        bs_tt[2].append(words[-1])
      elif "Testing" in words and "background" in words:
        bs_tt[3].append(words[-1])




   
rowcol = int(sqrt(nfiles)) + 1
c = TCanvas( "can", "can", 10, 10, 1200, 900)
c.SetFillColor(0)
c.Divide(rowcol,rowcol)

xaxistitle = [ "Training signal", "Training background", "Testing signal", "Testing background"] 
for f in range(0, nfiles):
  c.cd(f+1)
  for i in range(0, 4):
    for j in range(0, 4):
      if i==j:
        hnevents[f][i].Fill(xaxistitle[j], float(bs_tt[j][f]))
      else:
        hnevents[f][i].Fill(xaxistitle[j], 0.0)
    hnevents[f][i].SetFillColor(i+1)

    hnevents[f][i].GetXaxis().LabelsOption("v")

    hnevents[f][i].SetMinimum()
    if i==0:
      hnevents[f][i].Draw()
    else:
      hnevents[f][i].Draw("same")
  gPad.Update()




## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
                                                                                                                                                

