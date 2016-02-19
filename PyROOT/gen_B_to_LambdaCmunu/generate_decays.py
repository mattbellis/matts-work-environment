#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
from array import array

import numpy as np
import matplotlib.pylab as plt

from ROOT import TGenPhaseSpace,TLorentzVector,TRandom3,gROOT,gStyle

def energy(p3,mass):
    
    return np.sqrt(mass*mass + pmag([p3])**2)

def pmag(p3s):
    px,py,pz = 0,0,0
    for p3 in p3s:
        px += p3[0]
        py += p3[1]
        pz += p3[2]
    return np.sqrt(px*px + py*py + pz*pz)

def mass(p4s):
    e,px,py,pz = 0,0,0,0
    for p4 in p4s:
        e += p4[0]
        px += p4[1]
        py += p4[2]
        pz += p4[3]
    return np.sqrt(e*e - px*px - py*py - pz*pz)


batchMode = False

#
# Parse the command line options
#
max_events = int(sys.argv[1])

m_B = 5.279
m_LB = 5.619
m_Lc = 2.286
m_Dstarc = 2.010
m_Dstar0 = 2.007
m_Jpsi = 3.096

m_D0 = 1.864

m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511
m_nu = 0.0

#masses_sub = [ m_p, m_kc, m_pic, m_mu, m_e ]
rnd = TRandom3()

kaons = []
pions = []
muons = []
electrons = []
gammas = []

# B- --> A B C
masses0 = array('d', [ m_Lc, m_mu, m_nu ])
n0 = len(masses0)

masses1 = array('d', [ m_p, m_kc, m_pic ])
n1 = len(masses1)

#masses2 = array('d', [ m_kc, m_pic ])
#n2 = len(masses2)

resolution = 0.005 
#resolution = 0.00000001 
#
# Last argument determines batch mode or not
#
#last_argument = len(sys.argv) - 1
if (sys.argv[-1] == "batch"):
  batchMode = True

##############################################
# Start the generation
##############################################
event0 = TGenPhaseSpace()
event1 = TGenPhaseSpace()
event2 = TGenPhaseSpace()


################################################################################
################################################################################
def Bdecay():

    protons = []
    kaons = []
    pions = []
    muons = []
    electrons = []

    x = 50.*rnd.Rndm() - 25.
    y = 50.*rnd.Rndm() - 25.
    z = 50.*rnd.Rndm() - 25.
    initial_B = TLorentzVector();
    initial_B.SetXYZM(x,y,z,m_LB)

    weight = -999
    #print "here"
    if event0.SetDecay(initial_B, n0, masses0, ""):

        #print "Bdecay"

        # Decay the B
        weight = event0.Generate()
        #print weight

        # Grab the 4vecs for the Dstar and pi
        p_Lc = event0.GetDecay(0) # This returns a TLorentzVector
        p_mu = event0.GetDecay(1) # This returns a TLorentzVector
        p_nu = event0.GetDecay(2) # This returns a TLorentzVector

        muons.append([p_mu.E(),p_mu.Px(),p_mu.Py(),p_mu.Pz(),-1])
        electrons.append([p_nu.E(),p_nu.Px(),p_nu.Py(),p_nu.Pz(),0])

        # Decay the Lambda_c
        if event1.SetDecay(p_Lc, n1, masses1, ""):
            weight *= event1.Generate()

            p_p = event1.GetDecay(0) # This returns a TLorentzVector
            p_Kc = event1.GetDecay(1) # This returns a TLorentzVector
            p_pic = event1.GetDecay(2) # This returns a TLorentzVector

            protons.append([p_p.E(),p_p.Px(),p_p.Py(),p_p.Pz(),+1])
            kaons.append([p_Kc.E(),p_Kc.Px(),p_Kc.Py(),p_Kc.Pz(),-1])
            pions.append([p_pic.E(),p_pic.Px(),p_pic.Py(),p_pic.Pz(),+1])

    return weight,protons,kaons,pions,muons,electrons
################################################################################
################################################################################

# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0, 1000):
    # Keep track of the maximum weight for the event.
    weight,protons,kaons,pions,muons,electrons = Bdecay()
    if weight>maxweight and weight>0:
        maxweight = weight

print "maxweight: %f" % (maxweight)
     
# Generate the events!
outfilename = "ToyMC_LHCb_BtoLambdacmunu.dat"
#outfilename = "ToyMC_LHCb_BtoLambdacmunu_5MeV_resolution.dat"
outfilename = "ToyMC_LHCb_BtoLambdacmunu_0.5pct_resolution_100k.dat"
outfile = open(outfilename,'w')

nevents = 0
for i in range(0, max_events):

    if i%100==0:
        print i

    weight,protons,kaons,pions,muons,electrons = Bdecay()
    if weight<maxweight and weight>0:
        output = "Event: %d\n" % (nevents)

        output += "%d\n" % (len(pions))
        for p in pions:
            "---"
            #print p[0],p[1],p[2],p[3]
            p[1] += np.random.normal(loc=0.0,scale=resolution*abs(p[1]))
            p[2] += np.random.normal(loc=0.0,scale=resolution*abs(p[2]))
            p[3] += np.random.normal(loc=0.0,scale=resolution*abs(p[3]))
            p[0] = energy([p[1],p[2],p[3]],m_pic)
            #print p[0],p[1],p[2],p[3]
            output += "%f %f %f %f %d\n" % (p[0],p[1],p[2],p[3],p[4])

        output += "%d\n" % (len(kaons))
        for p in kaons:
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_kc)
            output += "%f %f %f %f %d\n" % (p[0],p[1],p[2],p[3],p[4])

        output += "%d\n" % (len(protons))
        for p in protons:
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_p)
            output += "%f %f %f %f %d\n" % (p[0],p[1],p[2],p[3],p[4])

        # Muons
        output += "%d\n" % (len(muons))
        for p in muons:
            "---"
            #print p[0],p[1],p[2],p[3]
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_mu)
            #print p[0],p[1],p[2],p[3]
            output += "%f %f %f %f %d\n" % (p[0],p[1],p[2],p[3],p[4])

        # Elecons ############ THESE WILL BE THE NEUTRON!
        #output += "%d\n" % (0)
        output += "%d\n" % (len(electrons))
        for p in electrons:
            "---"
            #print p[0],p[1],p[2],p[3]
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],0)
            #print p[0],p[1],p[2],p[3]
            output += "%f %f %f %f %d\n" % (p[0],p[1],p[2],p[3],p[4])

        # Photons
        output += "%d\n" % (0)

        outfile.write(output)

        nevents += 1


outfile.close()

################################################################################
## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
################################################################################
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
