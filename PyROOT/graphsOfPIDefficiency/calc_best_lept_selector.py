#!/usr/bin/env python

import sys
from math import *

sig = 100.0
bkg = 100.0
a = 5.0

lepton = sys.argv[1]
###############################################
# Open the input files
###############################################
infilename = [ '', '' ]
infile = []

infilename[0] = sys.argv[2] # Data
infilename[1] = sys.argv[3] # Sig MC

infile.append(open(infilename[0], "r"))
infile.append(open(infilename[1], "r"))
###############################################

sig = float(sys.argv[4])
bkg = float(sys.argv[5])
whichfom = sys.argv[6]

countmax = 8
if lepton == "mu":
  countmax = 8
elif lepton == "e":
  countmax = 6

nevents = []
for k in range(0,2):
  count = 0
  nevents.append([])
  for line in infile[k]:
    if count >= countmax:
      break
    vals = line.split()
    nevents[k].append( float(vals[2]) )

    #################################
    # Just grab the first few lines
    #################################
    count += 1



refindex = 4
if lepton == "mu":
  refindex = 4
elif lepton == "e":
  refindex = 0
reference = nevents[0][ refindex ]
#print nevents[0]
#print nevents[1]
######################
# Calc f.o.m.
######################
if whichfom == "0":
  print "Using S/sqrt(B)\t\tsig: %d\t\t bkg: %d" % (sig, bkg)
elif whichfom == "1":
  print "Using S/sqrt(S+B)\t\tsig: %d\t\t bkg: %d" % (sig, bkg)
elif whichfom == "2":
  print "Using S_eff/(sqrt(B*B_eff) + a/2\t\tsig: %d\t\t bkg: %d" % (sig, bkg)

for i,n in enumerate(nevents[0]):
  bkg_eff = n/reference
  sig_eff = nevents[1][i]/nevents[1][refindex]
  fom = 0
  if whichfom == "0":
    # S/sqrt(B)
    fom = sig_eff/sqrt(bkg_eff)
  elif whichfom == "1":
    # S/sqrt(S+B)
    fom = sig_eff*sig/sqrt(sig_eff*sig + bkg_eff*bkg)
  elif whichfom == "2":
    # Punzi: S_eff/(sqrt(B*B_eff) + a/2)
    fom = sig_eff/(sqrt(bkg_eff*bkg) + a/2.0)

  print "%3.3f %3.3f %3.3f" % (fom, sig_eff, bkg_eff)


