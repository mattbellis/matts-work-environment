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
mass_top = 173

m_q = 5

################################################################################
# Start with defining the inital state as a particle at rest
initial = TLorentzVector( 0.0, 0.0, 0.0, mass_top ) 
# Then get it moving
initial.SetXYZM(20, 40, 50, mass_top)

# Create an array of the masses of the final state particles
masses_f = array('d', [ m_q, m_q, m_q ])
n_f = len(masses_f)

##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()
rnd = TRandom3()

################################################################################
# Start generating the events
################################################################################
n = 1
while n < max_events:

    if event.SetDecay(initial, n_f, masses_f, ""):

        # Try generating an event
        weight = event.Generate()

        # Write out the total number of particles (initial + final)
        # and the 4vector of the initial state.
        print(n_f + 1)
        print("%f %f %f %f" % (initial.E(), initial.X(), initial.Y(), \
                initial.Z()))

        n += 1

        # Write out all the final state particles
        for i in range(0,n_f):
            particle = event.GetDecay(i)
            print("%f %f %f %f" % (particle.E(), particle.X(), particle.Y(), \
                    particle.Z()))


