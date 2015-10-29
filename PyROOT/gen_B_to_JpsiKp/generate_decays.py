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
m_Dstarc = 2.010
m_Dstar0 = 2.007
m_Jpsi = 3.096

m_D0 = 1.864

m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511

#masses_sub = [ m_p, m_kc, m_pic, m_mu, m_e ]
rnd = TRandom3()

kaons = []
pions = []
muons = []
electrons = []
gammas = []

# B- --> A B C
masses0 = array('d', [ m_Jpsi, m_kc, m_p ])
n0 = len(masses0)

masses1 = array('d', [ m_mu, m_mu ])
n1 = len(masses1)

masses2 = array('d', [ m_kc, m_pic ])
n2 = len(masses2)

resolution = 0.050 
#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
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

    kaons = []
    pions = []
    muons = []

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
        p_Jpsi = event0.GetDecay(0) # This returns a TLorentzVector
        p_Kc = event0.GetDecay(1) # This returns a TLorentzVector
        p_p = event0.GetDecay(2) # This returns a TLorentzVector

        kaons.append([p_Kc.E(),p_Kc.Px(),p_Kc.Py(),p_Kc.Pz(),-1,-999,-999,-999,-999,-999,-999,-999,-999])
        pions.append([p_p.E(),p_p.Px(),p_p.Py(),p_p.Pz(),+1,-999,-999,-999,-999,-999,-999,-999,-999])

        # Decay the J/psi
        if event1.SetDecay(p_Jpsi, n1, masses1, ""):
            weight *= event1.Generate()

            p_mu0 = event1.GetDecay(0) # This returns a TLorentzVector
            p_mu1 = event1.GetDecay(1) # This returns a TLorentzVector

            muons.append([p_mu0.E(),p_mu0.Px(),p_mu0.Py(),p_mu0.Pz(),+1,-999,-999,-999,-999,-999,-999,-999,-999])
            muons.append([p_mu1.E(),p_mu1.Px(),p_mu1.Py(),p_mu1.Pz(),-1,-999,-999,-999,-999,-999,-999,-999,-999])

    return weight,kaons,pions,muons
################################################################################
################################################################################

# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0, 1000):
    # Keep track of the maximum weight for the event.
    weight,kaons,pions,muons = Bdecay()
    if weight>maxweight and weight>0:
        maxweight = weight

print "maxweight: %f" % (maxweight)
     
# Generate the events!
outfilename = "ToyMC_LHCb_BtoJpsiKp.dat"
outfile = open(outfilename,'w')

nevents = 0
for i in range(0, max_events):

    if i%100==0:
        print i

    weight,kaons,pions,muons = Bdecay()
    if weight<maxweight and weight>0:
        output = "Event: %d\n" % (nevents)

        '''
        # Generate some random kaons and pions 
        nextra = np.random.randint(0,10)
        for j in range(nextra):
            x = 30.*rnd.Rndm() - 15
            y = 30.*rnd.Rndm() - 15
            z = 30.*rnd.Rndm() - 15
            e = energy([x,y,z],m_kc)
            q = 2*np.random.randint(0,2)-1
            kaons.append([e,x,y,z,q,-999,-999,-999,-999,-999,-999,-999,-999])
        
        nextra = np.random.randint(0,10)
        for j in range(nextra):
            x = 30.*rnd.Rndm() - 15
            y = 30.*rnd.Rndm() - 15
            z = 30.*rnd.Rndm() - 15
            e = energy([x,y,z],m_pic)
            q = 2*np.random.randint(0,2)-1
            pions.append([e,x,y,z,q,-999,-999,-999,-999,-999,-999,-999,-999])

        #print len(pions)

        np.random.shuffle(pions)
        np.random.shuffle(kaons)
        '''

        output += "%d\n" % (len(pions))
        for p in pions:
            "---"
            #print p[0],p[1],p[2],p[3]
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_p)
            #print p[0],p[1],p[2],p[3]
            output += "%f %f %f %f %d %f %f %f %f %d %d %f %f\n" % (p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])

        output += "%d\n" % (len(kaons))
        for p in kaons:
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_kc)
            output += "%f %f %f %f %d %f %f %f %f %d %d %f %f\n" % (p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])

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
            output += "%f %f %f %f %d %f %f %f %f %d %d %f %f\n" % (p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])

        # Elecons
        output += "%d\n" % (0)

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
