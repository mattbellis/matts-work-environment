import numpy as np
import matplotlib as plt

import ROOT

import sys

f = ROOT.TFile(sys.argv[1])

f.ls()

tree = f.Get("data")

tree.Print()

#exit()

nentries = tree.GetEntries()

outfilename = "%s.txt" % (sys.argv[1].split('/')[-1].split('.root')[0])
outfile = open(outfilename,'w')

for i in xrange(nentries):

    output = "Event: %d\n" % (i)
    tree.GetEntry(i)

    ############################################################################
    # Print out the jets
    ############################################################################
    njets = tree.NJet
    output += "%d\n" % (njets)
    for j in xrange(njets):
        e = tree.Jet_E[j]
        px = tree.Jet_Px[j]
        py = tree.Jet_Py[j]
        pz = tree.Jet_Pz[j]
        output += "%-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz)

        mass = np.sqrt(e**2 - (px**2 + py**2 + pz**2))
        print mass

    ############################################################################
    # Print out the muons
    ############################################################################
    nmuons = tree.NMuon
    output += "%d\n" % (nmuons)
    for j in xrange(nmuons):
        e = tree.Muon_E[j]
        px = tree.Muon_Px[j]
        py = tree.Muon_Py[j]
        pz = tree.Muon_Pz[j]
        output += "%-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz)

    ############################################################################
    # Print out the electrons
    ############################################################################
    nelectrons = tree.NElectron
    output += "%d\n" % (nelectrons)
    for j in xrange(nelectrons):
        e = tree.Electron_E[j]
        px = tree.Electron_Px[j]
        py = tree.Electron_Py[j]
        pz = tree.Electron_Pz[j]
        output += "%-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz)

    ############################################################################
    # Print out the photons
    ############################################################################
    nphotons = tree.NPhoton
    output += "%d\n" % (nphotons)
    for j in xrange(nphotons):
        e = tree.Photon_E[j]
        px = tree.Photon_Px[j]
        py = tree.Photon_Py[j]
        pz = tree.Photon_Pz[j]
        output += "%-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz)

    #print output
    outfile.write(output)

outfile.close()


