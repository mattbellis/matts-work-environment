import numpy as np
from numpy import dtype
import matplotlib as plt

import ROOT

import sys

f = ROOT.TFile(sys.argv[1])

f.ls()

tree = f.Get("data")

#tree.Print()

#exit()

nentries = tree.GetEntries()

outfilename = "%s_structured_array.npy" % (sys.argv[1].split('/')[-1].split('.root')[0])
#outfile = open(outfilename,'w')

events = []
print events
#exit()

for i in xrange(nentries):

    if i%100==0:
        print i

    #output = "Event: %d\n" % (i)
    output = ""
    tree.GetEntry(i)

    event = []

    nvals = 0
    ############################################################################
    # Print out the jets
    ############################################################################
    njets = tree.NJet
    vals = np.zeros(njets, dtype=dtype([('e', float),\
                                        ('px', float),\
                                        ('py', float),\
                                        ('pz', float),\
                                        ('pjet_btag', float)]))
    for j in xrange(njets):
        vals[j][0] = tree.Jet_E[j]
        vals[j][1] = tree.Jet_Px[j]
        vals[j][2] = tree.Jet_Py[j]
        vals[j][3] = tree.Jet_Pz[j]
        vals[j][4] = tree.Jet_btag[j]

    event.append(vals)

    ############################################################################
    # Print out the muons
    ############################################################################
    nmuons = tree.NMuon
    vals = np.zeros(nmuons, dtype=dtype([('e', float),\
                                        ('px', float),\
                                        ('py', float),\
                                        ('pz', float),\
                                        ('q', int)]))
    for j in xrange(nmuons):
        vals[j][0] = tree.Muon_E[j]
        vals[j][1] = tree.Muon_Px[j]
        vals[j][2] = tree.Muon_Py[j]
        vals[j][3] = tree.Muon_Pz[j]
        vals[j][4] = tree.Muon_Charge[j]

    event.append(vals)

    ############################################################################
    # Print out the electrons
    ############################################################################
    nelectrons = tree.NElectron
    vals = np.zeros(nelectrons, dtype=dtype([('e', float),\
                                        ('px', float),\
                                        ('py', float),\
                                        ('pz', float),\
                                        ('q', int)]))
    for j in xrange(nelectrons):
        vals[j][0] = tree.Electron_E[j]
        vals[j][1] = tree.Electron_Px[j]
        vals[j][2] = tree.Electron_Py[j]
        vals[j][3] = tree.Electron_Pz[j]
        vals[j][4] = tree.Electron_Charge[j]

    event.append(vals)

    ############################################################################
    # Print out the photons
    ############################################################################
    nphotons = tree.NPhoton
    vals = np.zeros(nphotons, dtype=dtype([('e', float),\
                                        ('px', float),\
                                        ('py', float),\
                                        ('pz', float)]))
    for j in xrange(nphotons):
        vals[j][0] = tree.Photon_E[j]
        vals[j][1] = tree.Photon_Px[j]
        vals[j][2] = tree.Photon_Py[j]
        vals[j][3] = tree.Photon_Pz[j]

    event.append(vals)

    vals = np.zeros(1, dtype=dtype([('px', float), ('py', float)]))
    vals[0][0] = tree.MET_px
    vals[0][1] = tree.MET_py

    event.append(vals)

    events.append(event)

np.save(outfilename,events)
