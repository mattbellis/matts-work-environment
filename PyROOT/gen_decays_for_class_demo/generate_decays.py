import os
import sys
from array import array

import ROOT 

batchMode = False

#masses_Xdecay = array('d', [ 0.10566, 0.10566 ])
#masses_Xdecay = array('d', [ 5.279, 5.279 ])
masses_Xdecay = array('d', [ 0.493677, 0.493677 ])
#masses_Xdecay = array('d', [ 0.13957, 0.13957 ])
n_Xdecay = len(masses_Xdecay)

##############################################
# Start the generation
##############################################
event = ROOT.TGenPhaseSpace()


initial_X = ROOT.TLorentzVector();
#initial_X.SetXYZM( 1.2, 0.5, 3.5, 3.096 ) 
#initial_X.SetXYZM( 0.0, 0.0, 0.0, 0.939 ) 
#initial_X.SetXYZM( 2.5, 3.6, 4.8, 0.497648 ) 
#initial_X.SetXYZM( 2.3, 3.1, 5.9, 10.564 ) 
initial_X.SetXYZM( 0.3, 1.2, 0.4, 1.019461 ) 

if event.SetDecay(initial_X, n_Xdecay, masses_Xdecay, ""):

    # Decay the B
    weight = event.Generate()
    print(weight)

    # Grab the 4vecs for the Lambda and mu
    p_A = event.GetDecay(0) # This returns a TLorentzVector
    p_B = event.GetDecay(1) # This returns a TLorentzVector
    #p_C = event.GetDecay(2) # This returns a TLorentzVector

    print("[{0}, {1}, {2}, {3}]".format(initial_X.T(),initial_X.X(),initial_X.Y(),initial_X.Z()))
    print(p_A.T(),p_A.X(),p_A.Y(),p_A.Z())
    print(p_B.T(),p_B.X(),p_B.Y(),p_B.Z())
    print("({0:.7f}, {1:.7f}, {2:.7f})".format(p_A.X(),p_A.Y(),p_A.Z()))
    print("({0:.7f}, {1:.7f}, {2:.7f})".format(p_B.X(),p_B.Y(),p_B.Z()))
    #print(p_C.T(),p_C.X(),p_C.Y(),p_C.Z())
