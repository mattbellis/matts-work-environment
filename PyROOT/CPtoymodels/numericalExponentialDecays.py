#!/usr/bin/env python

# Import the needed modules
import sys
import os

import array

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText
from ROOT import gROOT, gStyle, gPad, TLegend, TLine, TRandom3

batchMode = False

lifetime = float(sys.argv[1])
numpairs = int(sys.argv[2])
maxtime = float(sys.argv[3])
timestep = float(sys.argv[4])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(11);

h = []
h.append(TH1F("h0","Data",200,-20.0,20.0));
h.append(TH1F("h1","Data",200,-20.0,20.0));
h.append(TH1F("h2","Data",200,-20.0,20.0));

rnd = TRandom3()

pairs = [ ]
for i in range(0,numpairs):
  pairs.append([-1,-1])

# Number of time steps we will make
numsteps = int(maxtime/timestep)

# Make each step
for step in range(0,numsteps):
  t = step * timestep
  # Loop over each particle and pair
  for i in range(0,numpairs):
    for j in range(0,2):
      if pairs[i][j] < 0:
        prob_of_decay = rnd.Rndm()
        #print str(prob_of_decay) + " " + str(1.0/lifetime)
        if prob_of_decay < timestep*1.0/lifetime: 
          pairs[i][j] = t


numundecayed = 0
for i in range(0,numpairs):
  for j in range(0,2):
    h[j].Fill(pairs[i][j])
    if pairs[i][j] < 0:
      numundecayed += 1

  if ( pairs[i][0] >= 0.0 and pairs[i][j] >= 0.0):
    h[2].Fill( pairs[i][0] - pairs[i][1] )

print "numundecayed: " + str(numundecayed)
  
can = TCanvas("can","Data",10,10,1400,500)
can.SetFillColor(0)
can.Divide(3,1)


for j in range(0,3):
  can.cd(j+1)
  h[j].SetFillColor(43)
  if j==0:
    h[j].GetXaxis().SetTitle("t_{0}")
  elif j==1:
    h[j].GetXaxis().SetTitle("t_{1}")
  elif j==2:
    h[j].GetXaxis().SetTitle("#Delta t")
  h[j].Draw()
  gPad.Update()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



