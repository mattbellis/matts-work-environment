#!/usr/bin/env python

# Import the needed modules
import sys
import os

from array import array

from ROOT import *

batchMode = False

maxevents = int(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(11);

initial = TLorentzVector(0.0, 0.0, 0.0, 10.8)
masses = array('d', [0.139, 0.139, 0.139, 0.139, 0.139])

hmass = []
hmes = []
hmom = []
for i in range(0,10):
  name = "hmass" + str(i)
  hmass.append(TH1F(name,name,100,0.0, 10.0))

  name = "hmes" + str(i)
  hmes.append(TH1F(name,name,100,0.0, 10.0))

  name = "hmom" + str(i)
  hmom.append(TH1F(name,name,100,0.0, 6.0))

  hmass[i].SetFillColor(35)
  hmes[i].SetFillColor(25)
  hmom[i].SetFillColor(15)

event = TGenPhaseSpace()
rnd = TRandom3()


# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0,10000):
  if event.SetDecay(initial, 5, masses, ""):
    weight = event.Generate()
    if weight > maxweight:
      maxweight = weight

     

# Generate our events
pi = []
for i in range(0,5):
  pi.append(TLorentzVector())
n = 0
while n < maxevents:
  if event.SetDecay(initial, 5, masses, ""):
    weight = event.Generate()
    if maxweight*rnd.Rndm() < weight:
      #print n
      for i in range(0,5):
        pi[i] = event.GetDecay(i)
        #print pi[i].Rho()

      for i in range(0,4):
        for j in range(i+1,5):
          #print str(i) + str(j)
          twobody = pi[i] + pi[j]
          mass = twobody.M()
          mom = twobody.Rho()
          mes = sqrt(5.5*5.5 - twobody.Rho()*twobody.Rho())
          hmass[0].Fill(mass)
          hmes[0].Fill(mes)
          hmom[0].Fill(mom)

          k = int(mass) 
          index = k + 1
          if index<9:
            hmass[index].Fill(mass)
            hmes[index].Fill(mes)
            hmom[index].Fill(mom)



      n += 1


can = []
for i in range(0,1):
  name = "can"+str(i)
  can.append(TCanvas(name, name, 10+10*i, 10+10*i, 600,600))
  can[i].SetFillColor(0)
  can[i].Divide(3,9)

for i in range(0,1):
  for j in range(0,9):
    can[i].cd(j*3+1)
    hmass[j].Draw()
     
    can[i].cd(j*3+2)
    hmom[j].Draw()

    can[i].cd(j*3+3)
    hmes[j].Draw()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



