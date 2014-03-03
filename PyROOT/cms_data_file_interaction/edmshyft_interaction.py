#!/usr/bin/env python
# ==============================================================================
# ==============================================================================

from ROOT import gRandom, TH1, TH1D, cout, TCanvas
from ROOT import kRed,kBlack,kBlue

import ROOT

import sys

import numpy as np

# ==============================================================================
#  Example Unfolding
# ==============================================================================

# ==============================================================================
#  Read in the data.
# ==============================================================================

chain = ROOT.TChain("Events")

for file in sys.argv[1:]:
        chain.AddFile(file)

hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 40, 100,600)
hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 40, 100,600)
#hcsv = TH1D("CSV","CSV variable", 120, 0,1.2)
hcsv = []
for i in range(0,10):
    name = "hcsv%d" % (i)
    hcsv.append(TH1D(name,"CSV variable", 120, 0,1.2))

hnjets = TH1D("njets","njets", 10,-0.5,9.5)
htoppt = TH1D("toppt","toppt", 100,100.0,1100.0)

# Muon
muon_str = []
muon_str.append("floats_pfShyftTupleMuons_pt_ANA.obj")
muon_str.append("floats_pfShyftTupleMuons_eta_ANA.obj")
muon_str.append("floats_pfShyftTupleMuons_phi_ANA.obj")

# Top
top_str = []
top_str.append("floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj")
top_str.append("floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj")
top_str.append("floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj")
top_str.append("floats_pfShyftTupleJetsLooseTopTag_mass_ANA.obj")

# CSV jets
csvjet_str = []
csvjet_str.append("floats_pfShyftTupleJets_csv_ANA.obj")
csvjet_str.append("floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj")
csvjet_str.append("floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj")
csvjet_str.append("floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj")

#chain.SetBranchStatus('*', 1 )
chain.SetBranchStatus('*', 0 )
for s in muon_str:
    chain.SetBranchStatus(s, 1 )
for s in top_str:
    chain.SetBranchStatus(s, 1 )
for s in csvjet_str:
    chain.SetBranchStatus(s, 1 )

npossiblejets = 8

nev = chain.GetEntries()
print nev

'''
p4_meas.SetPtEtaPhiM(pt_meas,eta_meas,phi_meas,mass_meas);
#dR = np.sqrt((eta_meas-eta_truth)**2 + (phi_meas-phi_truth)**2)
dR0 = p4_meas.DeltaR(p4_truth);
'''


################################################################################
# Loop over the events
################################################################################
for n in xrange(nev):
    if n%10000==0:
        print "%d of %d" % (n,nev)

    chain.GetEntry(n)

    pt_meas = chain.GetLeaf(top_str[0]).GetValue(0)
    htoppt.Fill(pt_meas)

    #print "---------"
    njets = 0
    for i in xrange(npossiblejets):
        val = chain.GetLeaf(csvjet_str[0]).GetValue(i)
        if val>0.0:
            njets += 1
            #hcsv.Fill(val)
            for j in range(0,10):
                ptlo = 0 + j*100.0
                pthi = 0 + (j+1)*100.0
                if pt_meas>=ptlo and pt_meas<=pthi:
                    hcsv[j].Fill(val)
        #print val
    #print "njets: ",njets
    hnjets.Fill(njets)

################################################################################
# Unfold with one of the algorithms
################################################################################
c1 = TCanvas( 'c1', 'MC', 10, 10, 1400, 800 )
c1.Divide(2,1)
c1.cd(1)
hnjets.Draw()
c1.cd(2)
htoppt.Draw()

c2 = TCanvas( 'c2', 'MC', 20, 20, 1400, 800 )
c2.Divide(5,2)
for i in range(0,10):
    c2.cd(i+1)
    hcsv[i].Draw()

'''
hMC_true.SetLineColor(kBlack);  
hMC_true.Draw();  # MC raw 
c1.SaveAs("MC_true.png")
'''

#c1.Update()


################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

