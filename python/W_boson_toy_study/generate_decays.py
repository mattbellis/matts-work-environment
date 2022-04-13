import os
import sys
from array import array

import ROOT 
import time
import numpy as np

rnd = None
if rnd == None:
    rnd = ROOT.TRandom(int(1000000*time.time()))


batchMode = False
nevents = 1000

#masses_Xdecay = array('d', [ 0.10566, 0.10566 ])
#masses_Xdecay = array('d', [ 5.279, 5.279 ])
#masses_Xdecay = array('d', [ 0.493677, 0.493677 ])
masses_Xdecay = array('d', [ 0.105, 0.0 ])
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
#initial_X.SetXYZM( 0.3, 1.2, 0.4, 1.019461 ) 


output = "Ei,pxi,pyi,pzi,"
output += "E1,px1,py1,pz1,"
output += "E2,px2,py2,pz2\n"
print(output)
for i in range(nevents):

    mass = rnd.BreitWigner(83, 2.0)
    initial_X.SetXYZM( 0.0, 0.0, np.random.normal(10,2) , mass ) 

    if event.SetDecay(initial_X, n_Xdecay, masses_Xdecay, ""):

        #print("-------")
        # Decay the parent particle
        weight = event.Generate()
        #print(weight)

        # Grab the 4vecs for the Lambda and mu
        p_A = event.GetDecay(0) # This returns a TLorentzVector
        p_B = event.GetDecay(1) # This returns a TLorentzVector
        #p_C = event.GetDecay(2) # This returns a TLorentzVector

        output = ""
        output += f"{initial_X.T()},{initial_X.X()},{initial_X.Y()},{initial_X.Z()},"
        output += f"{p_A.T()},{p_A.X()},{p_A.Y()},{p_A.Z()},"
        output += f"{p_B.T()},{p_B.X()},{p_B.Y()},{p_B.Z()}\n"
        print(output)
        #print(p_C.T(),p_C.X(),p_C.Y(),p_C.Z())
