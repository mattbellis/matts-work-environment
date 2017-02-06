import numpy as np

from time import time

nevents = 100000
nparticles = 16
nentries  = 8

start = time()
muons = np.random.random((nevents,nparticles,nentries))
jets = np.random.random((nevents,nparticles,nentries))
electrons = np.random.random((nevents,nparticles,nentries))
dtime = time() - start
print "time to generate: %f s" % (dtime)

#muons[0][:,0] # Grab the first entries of the muons in the first event

################################################################################
import sys
objsize = (sys.getsizeof(muons) + sys.getsizeof(jets) + sys.getsizeof(electrons))/1e9 

print "size of arrays: %.3f Gb" % (objsize)
################################################################################

################################################################################
start = time()
'''
events = {"muons":muons,
          "jets":jets,
          "electrons":jets}
'''
events = zip(muons,jets,electrons)

dtime = time() - start
print "time to zip: %f s" % (dtime)

################################################################################

start = time()
masses = []
for event in events:

    ms,js,es = event

    for muon in ms:
        mass = (muon[0]**2 - muon[1]**2 - muon[2]**2 - muon[3]**2)
        masses.append(mass)


print len(masses)
dtime = time() - start
print "time to process events: %f s" % (dtime)

