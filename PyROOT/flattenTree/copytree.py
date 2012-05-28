#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import TBranch
from ROOT import gROOT, gStyle

gROOT.Reset()
#
# Parse the command line options
#
filename = sys.argv[1]

oldfile = TFile(filename)
oldtree = oldfile.Get("ntp1")
nentries = oldtree.GetEntries()
# Branches
nB = 0
new_nB = []
new_nB.append(0)
b_nB = TBranch()
#oldtree.SetBranchAddress("nB", nB, b_nB);


#Create a new file + a clone of old tree in new file
newfile = TFile("newfile.root","recreate")
newtree = oldtree.CloneTree(0)
#newtree.SetBranchAddress("new_nB",new_nB)
newtree.Branch('new_nB', 'new NB', new_nB, 32000, 99) 

for i in range(0, nentries):
  if i%100 == 0:
    print i
  oldtree.GetEntry(i)
  for j in range(0,oldtree.nB):
    print str(i) + " " + str(oldtree.nB)
    new_nB[0] = 4
    newtree.Fill()

#newtree.Print()
newtree.AutoSave()
newfile.Close()
newfile.Write()

##############################################


