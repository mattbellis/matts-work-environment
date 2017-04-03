import ROOT 
from array import array 
import numpy as np


# Create the file and "cd" into it.
f = ROOT.TFile("myrootfile.root", "RECREATE")
f.cd()

# Make the tree
tree = ROOT.TTree( 'T', 'My tree' )

# Make a branch to hold a single integer number per event
# Give it the default value of -1
nmuon = array('i', [-1])
tree.Branch('nmuon', nmuon, 'nmuon/I')

# Fill another value that depends on the number of muons (previous
# variable), but make it at least 16 entries long, per event.
muonpt = array('f', 16*[-1.])
tree.Branch('muonpt', muonpt, 'muonpt[nmuon]/F')

# Make some dummy events and entries
nevents = 1000

for i in range(0,nevents):

    # Generate a number between 0 and 16 "muons" 
    # to represent how many muons there are in this event.
    nmuon[0] = np.random.randint(0,17)

    # Generate random pt values
    for j in range(0,nmuon[0]):
        muonpt[j] = 200*np.random.random()


    # Must fill the tree
    tree.Fill()

# Go back "into" the file and close and write it.
f.cd()
f.Write()
f.Close()
