#!/usr/bin/env python

# Import the needed modules
import sys
import os

from array import array

from color_palette import *

from ROOT import *

batchMode = False

maxevents = int(sys.argv[1])

last_argument  = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gStyle.SetOptStat(11);
set_palette("palette",100)


photon = TLorentzVector(0.0, 0.0, 2.0, 2.0)
target = TLorentzVector(0.0, 0.0, 0.0, 0.938)
initial = photon + target
masses = array('d', [0.938, 0.139, 0.139])

#bvec = -1.0 * initial.BoostVector()
bvec = TVector3(-initial.BoostVector().X(), -initial.BoostVector().Y(), -initial.BoostVector().Z()) 

numcan = 9

hmass = []
for i in range(0,numcan):
  hmass.append([])
  for j in range(0,3):
    name = "hmass" + str(i) + "_" + str(j)
    if i==0:
      hmass[i].append(TH1F(name,name,100,0.0, 6.0))
    elif i==1:
      hmass[i].append(TH1F(name,name,100,-1.0, 3.0))
    elif i==2:
      hmass[i].append(TH1F(name,name,100,-1.1, 1.1))
    elif i==3:
      hmass[i].append(TH2F(name,name,100,-1.1, 1.0, 100, -1.0, 6.0))
    elif i==4:
      hmass[i].append(TH1F(name,name,100,-10.0, 10.0))
    elif i==5:
      hmass[i].append(TH2F(name,name,100,-10.0, 10.0, 100, -1.0, 6.0))
    elif i==6:
      hmass[i].append(TH2F(name,name,100,-10.0, 10.0, 100, -1.1, 1.1))
    elif i==7:
      hmass[i].append(TH2F(name,name,100,-3.0, 3.0, 100, -3.0, 3.0))
    elif i==8:
      hmass[i].append(TH2F(name,name,100,0.0, 4.2, 100, 0.0, 4.2))
    else:
      hmass[i].append(TH1F(name,name,100,0.0, 4.0))

    hmass[i][j].SetFillColor(35+i)
    hmass[i][j].SetMinimum(0.0)

event = TGenPhaseSpace()
rnd = TRandom3()


# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0,10000):
  if event.SetDecay(initial, 3, masses, ""):
    weight = event.Generate()
    if weight > maxweight:
      maxweight = weight

     

# Generate our events
m2 = [0.0, 0.0, 0.0]
pi = []
for i in range(0,5):
  pi.append(TLorentzVector())
n = 0
while n < maxevents:
  if event.SetDecay(initial, 3, masses, ""):
    weight = event.Generate()
    if maxweight*rnd.Rndm() < weight:
    #if 1.0:
      #print n
      for i in range(0,3):
        pi[i] = event.GetDecay(i)
        #print pi[i].Rho()

      m2[0] = (pi[1] + pi[2]).M2()
      m2[1] = (pi[0] + pi[2]).M2()
      m2[2] = (pi[0] + pi[1]).M2()

      for i in range(0,3):
        j=0
        k=0
        if i==0:
          j=1
          k=2
        elif i==1:
          j=0
          k=2
        elif i==2:
          j=0
          k=1
        #print str(n) + " " + str(j) + str(k)
        #print "\t" + str(pi[j].M()) + " " + str(pi[k].M())
        #print "\t" + str(m2[i]) 

        me = (photon - pi[j] + pi[k]).M2()

        t = (photon - pi[i]).M2()
        u = (target - pi[i]).M2()

        twobody = pi[j] + pi[k]
        mass = twobody.M2()
        #print "\t" + str(mass)

        cmvec = pi[i]
        #cmvec.Boost(bvec)
        cmct = cmvec.CosTheta()

        hmass[0][i].Fill(mass)
        hmass[1][i].Fill(-t)
        hmass[2][i].Fill(cmct)
        hmass[3][i].Fill(cmct, -t)
        hmass[4][i].Fill(me)
        hmass[5][i].Fill(me, -t)
        hmass[6][i].Fill(me, cmvec.Phi()/3.14159)
        hmass[7][i].Fill(-t, -u)
        hmass[8][i].Fill(m2[j], m2[k])

      n += 1


can = []
for i in range(0,numcan):
  name = "can"+str(i)
  can.append(TCanvas(name, name, 10+50*i, 10+50*i, 900, 300))
  can[i].SetFillColor(0)
  can[i].Divide(3,1)

for i in range(0,numcan):
  for j in range(0,3):
    can[i].cd(j+1)
    hmass[i][j].Draw()
    if i==3 or i==5 or i==6 or i==7 or i==8:
      hmass[i][j].Draw("colz")
    gPad.Update()
     

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]



