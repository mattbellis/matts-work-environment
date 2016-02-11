#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
from array import array

from ROOT import *

batchMode = False

masses_Xdecay = array('d', [ 0.105, 0.105 ])
n_Xdecay = len(masses_Xdecay)

##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()


initial_X = TLorentzVector();
initial_X.SetXYZM( 2.0, 3.0, 5.0, 3.096 ) 

if event.SetDecay(initial_X, n_Xdecay, masses_Xdecay, ""):

    # Decay the B
    weight = event.Generate()
    print weight

    # Grab the 4vecs for the Lambda and mu
    p_A = event.GetDecay(0) # This returns a TLorentzVector
    p_B = event.GetDecay(1) # This returns a TLorentzVector

    print initial_X.T(),initial_X.X(),initial_X.Y(),initial_X.Z()
    print p_A.T(),p_A.X(),p_A.Y(),p_A.Z()
    print p_B.T(),p_B.X(),p_B.Y(),p_B.Z()
