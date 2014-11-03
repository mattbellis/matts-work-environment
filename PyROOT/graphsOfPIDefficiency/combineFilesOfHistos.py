#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import *

#
# Parse the command line options
#
filenames = []
fileweights = []

newfile = sys.argv[1]
for i in range(2, len(sys.argv)):
  if i%2==0:
    filenames.append(sys.argv[i])
  else:
    fileweights.append(float(sys.argv[i]))

numfiles = len(filenames) 
numweights = len(fileweights)
if numfiles != numweights:
  print "Not same number of files and weights!"
  sys.exit(-1)

###################################
# Create the new file and histos
###################################
newrootfile = TFile(newfile, "RECREATE")

###################################
# Read in the files
###################################

can = TCanvas("c1","c1",10,10,600,600)
can.SetFillColor(0)
can.Divide(2,1)


hnew = []
rootfiles = []
for f in range(0, numfiles):
  print "Trying to open " + filenames[f]
  rootfiles.append(TFile(filenames[f], "READ"))
  hcount = 0
  for i in range(0,1):
    for j in range(0,3):
      #for k in range(0,219):
      for k in range(0,48):
      #for k in range(0,11):
        hname = "hmass" + str(i) + "_" + str(j) + "_" + str(k)
        hdum = gROOT.FindObject(hname) 
        #print "Looking for " + hname
        if (hdum):
          print "\tFound " + hname
          can.cd(1)
          hdum.Draw()
          gPad.Update()

          if f==0:
            print hdum.Integral()
            hnew.append(hdum)
            hnew[hcount].Scale(fileweights[f])
            print str(f) + " " + str(hnew[hcount].GetMaximum())
          else:
            print hdum.Integral()
            print hcount
            hnew[hcount].Add(hdum, fileweights[f])
            print str(f) + " " + str(hnew[hcount].GetMaximum())

          can.cd(2)
          hnew[hcount].Draw()
          gPad.Update()
          hcount+=1
        else:
          hnew.append(TH1F())
        

  #rootfiles[f].Close()


newrootfile.cd()
for i in range(0,hcount):
  hnew[i].Write()

newrootfile.Close()
newrootfile.Write()






