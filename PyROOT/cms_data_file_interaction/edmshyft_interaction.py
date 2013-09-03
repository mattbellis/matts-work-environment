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
str_truth_pt = "floats_pfShyftTupleMuonsLoose_pt.obj"
str_truth_eta = "floats_pfShyftTupleMuonsLoose_eta_ANA.obj"
str_truth_phi = "floats_pfShyftTupleMuonsLoose_phi_ANA.obj"

# Top
str_truth_pt = "floats_pfShyftTupleTopQuarks_pt_ANA.obj"
str_truth_eta = "floats_pfShyftTupleTopQuarks_eta_ANA.obj"
str_truth_phi = "floats_pfShyftTupleTopQuarks_phi_ANA.obj"

# CSV jets
str_csv = "floats_pfShyftTupleJets_csv_ANA.obj"
str_meas_pt = "floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj"
str_meas_eta = "floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj"
str_meas_phi = "floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj"

chain.SetBranchStatus('*', 1 )

npossiblejets = 8

nev = chain.GetEntries()
print nev

################################################################################
# Loop over the events
################################################################################
for n in xrange(nev):
    if n%10000==0:
        print "%d of %d" % (n,nev)

    chain.GetEntry(n)

    pt_meas = chain.GetLeaf(str_meas_pt).GetValue(0)
    htoppt.Fill(pt_meas)

    #print "---------"
    njets = 0
    for i in xrange(npossiblejets):
        val = chain.GetLeaf(str_csv).GetValue(i)
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
    '''
    # Loop over the top and antitop truth info.
    for i in range(0,2):

        pt_truth = chain.GetLeaf(str_truth_pt).GetValue(i)
        eta_truth = chain.GetLeaf(str_truth_eta).GetValue(i)
        phi_truth = chain.GetLeaf(str_truth_phi).GetValue(i)

        hMC_true.Fill(pt_truth)

        found_match = False
        min_dR = 100000.0
        min_dR_pt = -1
        min_index = -1

        # Loop over (if there are any) the reconstructed top jets.
        for j in range(0,2):

            pt_meas = chain.GetLeaf(str_meas_pt).GetValue(j)
            eta_meas = chain.GetLeaf(str_meas_eta).GetValue(j)
            phi_meas = chain.GetLeaf(str_meas_phi).GetValue(j)

            # Calc dR between truth and reconstructed jet.
            dR = np.sqrt((eta_meas-eta_truth)**2 + (phi_meas-phi_truth)**2)

            # Make some pt cuts for now. May relax this. 
            # If reconstructed is within dR<0.8, this is a match!
            if pt_meas > 100 and pt_meas<600:
                if dR < min_dR and dR < 0.8:
                    min_dR = dR
                    min_dR_pt = pt_meas
                    found_match = True

        # Fill the response matrix accordingly.
        if found_match == False:
            response.Miss (pt_truth)
        else:
            hMC_meas.Fill(min_dR_pt)
            response.Fill (min_dR_pt, pt_truth)


    '''

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

