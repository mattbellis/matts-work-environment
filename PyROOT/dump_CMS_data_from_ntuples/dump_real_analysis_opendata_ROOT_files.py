import numpy as np
import matplotlib as plt

import ROOT

import zipfile

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
ak5jet_str.append("recoPFJets_ak5PFJets__RECO.obj.pt_")
ak5jet_str.append("recoPFJets_ak5PFJets__RECO.obj.eta_")
ak5jet_str.append("recoPFJets_ak5PFJets__RECO.obj.phi_")
ak5jet_str.append("recoPFJets_ak5PFJets__RECO.obj.mass_")
#ak5jet_str.append("floats_pfShyftTupleJetsLooseAK5_csv_ANA.obj")

'''
ca8jet_str = []
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_mass_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_nSubjets_ANA.obj")
ca8jet_str.append("floats_pfShyftTupleJetsLooseTopTag_minMass_ANA.obj")
'''

muons_str = []
muons_str.append("recoMuons_muons__RECO.obj.pt_")
muons_str.append("recoMuons_muons__RECO.obj.eta_")
muons_str.append("recoMuons_muons__RECO.obj.phi_")
muons_str.append("recoMuons_muons__RECO.obj.qx3_") # What is this????

electrons_str = []
electrons_str.append("recoPFCandidates_particleFlow_electrons_RECO.obj.pt_")
electrons_str.append("recoPFCandidates_particleFlow_electrons_RECO.obj.eta_")
electrons_str.append("recoPFCandidates_particleFlow_electrons_RECO.obj.phi_")
electrons_str.append("recoPFCandidates_particleFlow_electrons_RECO.obj.qx3_")

photons_str = []
photons_str.append("recoPhotons_photons__RECO.obj.pt_")
photons_str.append("recoPhotons_photons__RECO.obj.eta_")
photons_str.append("recoPhotons_photons__RECO.obj.phi_")

met_str = []
met_str.append("recoPFMETs_pfMet__RECO.obj.pt_")
met_str.append("recoPFMETs_pfMet__RECO.obj.phi_")
'''
'''


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
#for s in ca8jet_str:
    #chain.SetBranchStatus(s, 1 )
for s in muons_str:
    chain.SetBranchStatus(s, 1 )
for s in electrons_str:
    chain.SetBranchStatus(s, 1 )
for s in photons_str:
    chain.SetBranchStatus(s, 1 )
for s in met_str:
    chain.SetBranchStatus(s, 1 )


#exit()

nentries = chain.GetEntries()

#outfilename = "for_analysis_%s.txt" % (sys.argv[1].split('/')[-1].split('.root')[0])
outfilename = "%s_for_analysis.txt" % (sys.argv[1].split('.root')[0])
outfile = open(outfilename,'w')

for i in xrange(nentries):

    if i%1000==0:
        print "Event: %d" % (i)

    output = "Event: %d\n" % (i)
    #output = ""
    chain.GetEntry(i)

    ############################################################################
    # Print out the not-top jets
    ############################################################################
    nthings = 0
    temp_output = ""
    for i in xrange(32):
        pt = chain.GetLeaf(ak5jet_str[0]).GetValue(i)
        eta = chain.GetLeaf(ak5jet_str[1]).GetValue(i)
        phi = chain.GetLeaf(ak5jet_str[2]).GetValue(i)
        mass = chain.GetLeaf(ak5jet_str[3]).GetValue(i)
        #csv = chain.GetLeaf(ak5jet_str[4]).GetValue(i)
        csv = -999.
        if mass==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        #print nthings
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz,csv)
        #print pt,eta,phi,mass,csv

    #print nthings
    output += "%d\n%s" % (nthings,temp_output)

    ############################################################################
    # Print out the top jets
    ############################################################################
    output += "%d\n" % (0)
    '''
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
    '''

    ############################################################################
    # Print out the muons
    ############################################################################
    nthings = 0
    temp_output = ""
    mass = 0.105
    for i in xrange(16):
        pt = chain.GetLeaf(muons_str[0]).GetValue(i)
        eta = chain.GetLeaf(muons_str[1]).GetValue(i)
        phi = chain.GetLeaf(muons_str[2]).GetValue(i)
        q = chain.GetLeaf(muons_str[3]).GetValue(i)
        if pt==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz,q)
        #print pt,eta,phi

    output += "%d\n%s" % (nthings,temp_output)

    ############################################################################
    # Print out the electrons
    ############################################################################
    nthings = 0
    temp_output = ""
    mass = 0.000511
    for i in xrange(16):
        pt = chain.GetLeaf(electrons_str[0]).GetValue(i)
        eta = chain.GetLeaf(electrons_str[1]).GetValue(i)
        phi = chain.GetLeaf(electrons_str[2]).GetValue(i)
        q = chain.GetLeaf(electrons_str[3]).GetValue(i)
        if pt==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz,q)
        #print pt,eta,phi

    output += "%d\n%s" % (nthings,temp_output)

    ############################################################################
    # Print out the photons
    ############################################################################
    nthings = 0
    temp_output = ""
    mass = 0.0
    for i in xrange(16):
        pt = chain.GetLeaf(photons_str[0]).GetValue(i)
        eta = chain.GetLeaf(photons_str[1]).GetValue(i)
        phi = chain.GetLeaf(photons_str[2]).GetValue(i)
        if pt==0.0:
            break
        nthings += 1

        px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        temp_output += "%-10.4f %-10.4f %-10.4f %-10.4f\n" % (e,px,py,pz)
        #print pt,eta,phi

    output += "%d\n%s" % (nthings,temp_output)


    ############################################################################
    # Print out the MET
    ############################################################################
    nthings = 0
    temp_output = ""
    for i in xrange(1):
        pt = chain.GetLeaf(met_str[0]).GetValue(i)
        phi = chain.GetLeaf(met_str[1]).GetValue(i)
        if pt==0.0:
            break
        nthings += 1

        #px,py,pz = ptetaphi_to_xyz(pt,eta,phi)
        #e = np.sqrt(mass*mass + px*px + py*py + pz*pz)
        
        temp_output += "%-10.4f %-10.4f\n" % (pt,phi)
        #print pt,eta,phi

    output += "%d\n%s" % (nthings,temp_output)
    '''
    '''

    #print output
    outfile.write(output)



outfile.close()

#zipfilename = "%s.zip" % (sys.argv[1].split('/')[-1].split('.root')[0])
zipfilename = "%s.zip" % (sys.argv[1].split('.root')[0])
zf = zipfile.ZipFile(zipfilename,'w')
zf.write(outfilename,compress_type=zipfile.ZIP_DEFLATED)
zf.close()
#outfile = open(outfilename,'w')
