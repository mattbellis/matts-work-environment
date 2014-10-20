import numpy as np
import matplotlib as plt

import ROOT

import sys

def ptetaphi_to_xyz(pt,eta,phi):
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)

    return px,py,pz
    
# This is from Louise
# The AK5GenJets are truth-level jets that were formed by running the 
# AK5 jet algorithm at truth level, and similarly the CA8 jets. 
#
# AK5 jets are what we use for the "regular" jets while the 
# CA8 jets are the Cambridge-Achen dR=0.8, are the input to the top-tagged jets.

ak5jet_str = []
ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_pt_ANA.obj")
ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_eta_ANA.obj")
ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_phi_ANA.obj")
ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_mass_ANA.obj")
ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_csv_ANA.obj")

ca8jet_str = []
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_mass_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_nSubjets_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_minMass_ANA.obj")


chain = ROOT.TChain("Events")

for file in sys.argv[1:]:
    chain.AddFile(file)



#f = ROOT.TFile(sys.argv[1])
#f.ls()
#tree = f.Get("data")
#tree = chain.Get("data")
#tree.Print()

chain.SetBranchStatus('*', 0 )
for s in ak5jet_str:
    chain.SetBranchStatus(s, 1 )
for s in ca8jet_str:
    chain.SetBranchStatus(s, 1 )


#exit()

nentries = chain.GetEntries()

#outfilename = "for_analysis_%s.txt" % (sys.argv[1].split('/')[-1].split('.root')[0])
outfilename = "%s_for_analysis.txt" % (sys.argv[1].split('.root')[0])
outfile = open(outfilename,'w')

for i in xrange(nentries):

    output = "Event: %d\n" % (i)
    #output = ""
    chain.GetEntry(i)

    ############################################################################
    # Print out the not-top jets
    ############################################################################
    nthings = 0
    temp_output = ""
    for i in xrange(16):
        pt = chain.GetLeaf(ak5jet_str[0]).GetValue(i)
        eta = chain.GetLeaf(ak5jet_str[1]).GetValue(i)
        phi = chain.GetLeaf(ak5jet_str[2]).GetValue(i)
        mass = chain.GetLeaf(ak5jet_str[3]).GetValue(i)
        csv = chain.GetLeaf(ak5jet_str[4]).GetValue(i)
        if csv==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        print nthings
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz,csv)
        print pt,eta,phi,mass,csv

    print nthings
    output += "%d\n%s" % (nthings,temp_output)

    ############################################################################
    # Print out the top jets
    ############################################################################
    nthings = 0
    temp_output = ""
    for i in xrange(16):
        pt = chain.GetLeaf(ca8jet_str[0]).GetValue(i)
        eta = chain.GetLeaf(ca8jet_str[1]).GetValue(i)
        phi = chain.GetLeaf(ca8jet_str[2]).GetValue(i)
        mass = chain.GetLeaf(ca8jet_str[3]).GetValue(i)
        nsub = chain.GetLeaf(ca8jet_str[4]).GetValue(i)
        minmass = chain.GetLeaf(ca8jet_str[5]).GetValue(i)
        if pt==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz,nsub,minmass)
        print pt,eta,phi,mass,nsub,minmass

    output += "%d\n%s" % (nthings,temp_output)

    print output
    outfile.write(output)

outfile.close()


