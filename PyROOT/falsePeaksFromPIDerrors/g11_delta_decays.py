#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
from array import array

from matplotlib.font_manager import FontProperties

import matplotlib.pylab as plt
import numpy as np

from ROOT import TGenPhaseSpace,TLorentzVector,TRandom3,TVector3

from color_palette import *
import lichen.lichen as lch

batchMode = False

#
# Parse the command line options
#
max_events = float(sys.argv[1])

#mass_i = 5.279
mass_i = 2.289

m_lam = 1.115
m_delta = 1.232
m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511

beam = TLorentzVector(0.0,0.0,4.0,4.0)
target = TLorentzVector( 0.0,0.0,0.0, 0.938)
#initial = TLorentzVector( px,py,pz, sqrt(pmag2+m_lam*m_lam) ) 
initial = beam + target

masses_f = array('d', [ m_pic, m_delta ])
n_f = len(masses_f)

masses_lam_decay = array('d', [ m_p, m_pic ])
n_lam_decay = len(masses_lam_decay)

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()
lambda_decay = TGenPhaseSpace()
rnd = TRandom3()

#bvec = -1.0 * initial.BoostVector()
bvec = TVector3(-initial.BoostVector().X(), -initial.BoostVector().Y(), -initial.BoostVector().Z()) 

# Calculate the max weight for this topology
maxweight = 0.0
maxweight_lam = 0.0
for i in range(0, 10000):

    masses_f[1] = m_delta + rnd.Gaus(0.0,0.060)
    if event.SetDecay(initial, n_f, masses_f, ""):
        weight = event.Generate()
        if weight > maxweight:
            maxweight = weight
    
        lambda_decay.SetDecay(event.GetDecay(1),n_lam_decay,masses_lam_decay,"")
        lam_weight = lambda_decay.Generate()
        if lam_weight > maxweight_lam:
            maxweight_lam = lam_weight

    '''
    print "---------- WEIGHT ---------"
    print event.GetDecay(0).M()
    event.GetDecay(0).Print()
    print lambda_decay.GetDecay(0).M()
    lambda_decay.GetDecay(0).Print()
    print lambda_decay.GetDecay(1).M()
    lambda_decay.GetDecay(1).Print()
    '''

#exit()

     

num_combos = 1
# Generate our events
m2 = [0.0, 0.0, 0.0]
p_f = []
for i in range(0, n_f):
  p_f.append(TLorentzVector())

n = 0
pres = 0.005
lam_masses = []
lam_masses_false = []

mm_masses = []
mm_masses_false = []

while n < max_events:
    masses_f[1] = m_delta + rnd.Gaus(0.0,0.060)
    #print masses_f
    if event.SetDecay(initial, n_f, masses_f, ""):

      weight = event.Generate()
  
      lambda_decay.SetDecay(event.GetDecay(1),n_lam_decay,masses_lam_decay,"")
      lam_weight = lambda_decay.Generate()
  
      if maxweight*rnd.Rndm() < weight and maxweight_lam*rnd.Rndm()<lam_weight:
  
          kaon = event.GetDecay(0)
          proton = lambda_decay.GetDecay(0)
          pion = lambda_decay.GetDecay(1)
  
          #print "---------- events ---------"
          #kaon.Print()
          #proton.Print()
          #pion.Print()

          for m,p in zip([m_pic,m_p,m_pic],[kaon,proton,pion]):
              p.SetX(p.X() + rnd.Gaus(0,pres))
              p.SetY(p.Y() + rnd.Gaus(0,pres))
              p.SetZ(p.Z() + rnd.Gaus(0,pres))
              p.SetE(np.sqrt(p.P()*p.P() + m*m))
              #p.Print()

          lam_masses.append((proton+pion).M())
          mm_masses.append((initial-kaon).M())

          # False lambdas
          for m,p in zip([m_kc,m_p],[kaon,proton]):
              p.SetE(np.sqrt(p.P()*p.P() + m*m))
              #p.Print()
  
          lam_masses_false.append((proton+pion).M())
          mm_masses_false.append((initial-kaon).M())

          if n%1000 == 0:
            print n
  
          n += 1


#print lam_masses

print len(lam_masses)
print len(mm_masses)
print len(lam_masses_false)
print len(mm_masses_false)

font0 = FontProperties()
font0.set_family('serif')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
lch.hist_err(lam_masses,bins=100,range=(0.9,1.5))
plt.xlabel(r'True $\Delta^0$ masses',fontproperties=font0)
plt.subplot(1,2,2)
lch.hist_err(mm_masses,bins=100,range=(0.9,1.5))
plt.xlabel(r'True missing masses off $\pi^+$',fontproperties=font0)
plt.savefig('Delta_1D.png')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
lch.hist_err(lam_masses_false,bins=100,range=(0.9,1.5))
plt.xlabel(r'False $\Lambda$ masses',fontproperties=font0)
plt.subplot(1,2,2)
lch.hist_err(mm_masses_false,bins=100,range=(0.9,1.5))
plt.xlabel(r'False missing masses off $K^+$',fontproperties=font0)
plt.savefig('Fake_Lambda_1D.png')

plt.figure(figsize=(10,5))
lch.hist_2D(lam_masses,mm_masses,xbins=100,ybins=50,xrange=(0.9,1.5),yrange=(0.9,1.5))
plt.xlabel(r'True $\Delta^0$ masses',fontproperties=font0)
plt.ylabel(r'True missing masses off $\pi^+$',fontproperties=font0)
plt.savefig('Delta_2D.png')

plt.figure(figsize=(10,5))
lch.hist_2D(lam_masses_false,mm_masses_false,xbins=100,ybins=50,xrange=(0.9,1.5),yrange=(0.9,1.5))
plt.xlabel(r'False $\Lambda$ masses',fontproperties=font0)
plt.ylabel(r'False missing masses off $K^+$',fontproperties=font0)
plt.savefig('Fake_Lambda_2D.png')

plt.show()

################################################################################
