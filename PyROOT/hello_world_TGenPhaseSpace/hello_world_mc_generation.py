#!/usr/bin/env python
#

################################################################################
# Import the needed modules
################################################################################
import os
import sys
from array import array

from ROOT import *

################################################################################
# Parse the command line options
################################################################################
max_events = float(sys.argv[1])

################################################################################
# Masses of particles
################################################################################
mass_B = 5.279
mass_lc = 2.289

m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511

################################################################################
# Start with defining the inital state as a particle at rest
initial = TLorentzVector( 0.0, 0.0, 0.0, mass_lc ) 

# Create an array of the masses of the final state particles
masses_f = array('d', [ m_p, m_kc, m_pic ])
n_f = len(masses_f)

##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()
rnd = TRandom3()

################################################################################
# Start generating the events
################################################################################
n = 0
while n < max_events:

    if event.SetDecay(initial, n_f, masses_f, ""):

        # Try generating an event
        weight = event.Generate()

        # Write out the total number of particles (initial + final)
        # and the 4vector of the initial state.
        print n_f + 1
        print "%f %f %f %f" % (initial.E(), initial.X(), initial.Y(), \
                initial.Z())

        n += 1

        # Write out all the final state particles
        for i in range(0,n_f):
            particle = event.GetDecay(i)
            print "%f %f %f %f" % (particle.E(), particle.X(), particle.Y(), \
                    particle.Z())


