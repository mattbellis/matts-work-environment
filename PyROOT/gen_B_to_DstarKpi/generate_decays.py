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
m_Dstarc = 2.010
m_Dstar0 = 2.007

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

# B- --> D*+ K- pi-
masses_Bdecay = array('d', [ m_Dstarc, m_kc, m_pic ])
n_Bdecay = len(masses_Bdecay)

masses_Dstarcdecay = array('d', [ m_D0, m_pic ])
n_Dstarcdecay = len(masses_Dstarcdecay)

masses_D0decay = array('d', [ m_kc, m_pic ])
n_D0decay = len(masses_D0decay)

resolution = 0.00000100 
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
event_Dstar = TGenPhaseSpace()
event_D = TGenPhaseSpace()


################################################################################
################################################################################
def Bdecay():

    kaons = []
    pions = []

    x = 50.*rnd.Rndm() - 25.
    y = 50.*rnd.Rndm() - 25.
    z = 50.*rnd.Rndm() - 25.
    initial_B = TLorentzVector();
    initial_B.SetXYZM(x,y,z,m_B)

    weight = -999
    #print "here"
    if event.SetDecay(initial_B, n_Bdecay, masses_Bdecay, ""):

        #print "Bdecay"

        # Decay the B
        weight = event.Generate()
        #print weight

        # Grab the 4vecs for the Dstar and pi
        p_Dstar = event.GetDecay(0) # This returns a TLorentzVector
        p_Kc = event.GetDecay(1) # This returns a TLorentzVector
        p_pic = event.GetDecay(2) # This returns a TLorentzVector

        kaons.append([p_Kc.E(),p_Kc.Px(),p_Kc.Py(),p_Kc.Pz(),-1,-999,-999,-999,-999,-999,-999,-999,-999])
        pions.append([p_pic.E(),p_pic.Px(),p_pic.Py(),p_pic.Pz(),-1,-999,-999,-999,-999,-999,-999,-999,-999])

        # Decay the D*
        if event_Dstar.SetDecay(p_Dstar, n_Dstarcdecay, masses_Dstarcdecay, ""):
            weight *= event_Dstar.Generate()

            p_D0Dstar = event_Dstar.GetDecay(0) # This returns a TLorentzVector
            p_pic = event_Dstar.GetDecay(1) # This returns a TLorentzVector

            pions.append([p_pic.E(),p_pic.Px(),p_pic.Py(),p_pic.Pz(),+1,-999,-999,-999,-999,-999,-999,-999,-999])

            # Decay the D
            if event_D.SetDecay(p_D0Dstar, n_D0decay, masses_D0decay, ""):
                weight *= event_D.Generate()

                p_Kc = event_D.GetDecay(0) # This returns a TLorentzVector
                p_pic = event_D.GetDecay(1) # This returns a TLorentzVector

                kaons.append([p_Kc.E(),p_Kc.Px(),p_Kc.Py(),p_Kc.Pz(),-1,-999,-999,-999,-999,-999,-999,-999,-999])
                pions.append([p_pic.E(),p_pic.Px(),p_pic.Py(),p_pic.Pz(),+1,-999,-999,-999,-999,-999,-999,-999,-999])

    return weight,kaons,pions
################################################################################
################################################################################

# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0, 1000):
    # Keep track of the maximum weight for the event.
    weight,kaons,pions = Bdecay()
    if weight>maxweight and weight>0:
        maxweight = weight

print "maxweight: %f" % (maxweight)
     
# Generate the events!
#outfilename = "ToyMC_LHCb_BtoDstarKpi_smeared.dat"
outfilename = "ToyMC_LHCb_BtoDstarKpi.dat"
outfile = open(outfilename,'w')

nevents = 0
for i in range(0, max_events):

    weight,kaons,pions = Bdecay()
    if weight<maxweight and weight>0:
        output = "Event: %d\n" % (nevents)

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

        output += "%d\n" % (len(pions))
        for p in pions:
            "---"
            print p[0],p[1],p[2],p[3]
            p[1] += np.random.normal(loc=0.0,scale=resolution)
            p[2] += np.random.normal(loc=0.0,scale=resolution)
            p[3] += np.random.normal(loc=0.0,scale=resolution)
            p[0] = energy([p[1],p[2],p[3]],m_pic)
            print p[0],p[1],p[2],p[3]
            output += "%f %f %f %f %d %f %f %f %f %d %d %f %f\n" % (p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])

        output += "%d\n" % (len(kaons))
        for p in kaons:
            output += "%f %f %f %f %d %f %f %f %f %d %d %f %f\n" % (p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])

        # Muons
        output += "%d\n" % (0)

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
