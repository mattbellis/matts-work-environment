#!/usr/bin/env python
# ==============================================================================
# ==============================================================================

from ROOT import gRandom, TH1, TH1D, cout, TCanvas
from ROOT import TLorentzVector
from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes
from ROOT import kRed,kBlack,kBlue
from ROOT import RooUnfoldSvd
#from ROOT import RooUnfoldTUnfold

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

print "==================================== TRAIN ===================================="
#response= RooUnfoldResponse (40,100,600)
#hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 40, 100,600)
#hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 40, 100,600)

response= RooUnfoldResponse (10,100,1100)
hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 10,100,1100)
hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 10,100,1100)

# Truth?
str_truth_pt = "floats_pfShyftTupleTopQuarks_pt_ANA.obj"
str_truth_eta = "floats_pfShyftTupleTopQuarks_eta_ANA.obj"
str_truth_phi = "floats_pfShyftTupleTopQuarks_phi_ANA.obj"
str_truth_mass = "floats_pfShyftTupleTopQuarks_mass_ANA.obj"

# Reconstructed
str_meas_pt = "floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj"
str_meas_eta = "floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj"
str_meas_phi = "floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj"
str_meas_mass = "floats_pfShyftTupleJetsLooseTopTag_mass_ANA.obj"

chain.SetBranchStatus('*', 1 )

# Assume for now that there are only two tops.
ntops = 2

nev = chain.GetEntries()
print nev

pt_truth = [0.0,0.0]
pt_meas = [0.0,0.0]

eta_truth = [0.0,0.0]
eta_meas = [0.0,0.0]

phi_truth = [0.0,0.0]
phi_meas = [0.0,0.0]

mass_truth = [0.0,0.0]
mass_meas = [0.0,0.0]

p4_truth = TLorentzVector()
p4_meas = TLorentzVector()

################################################################################
# Loop over the events
################################################################################
for n in xrange(nev):
    if n%10000==0:
        print "%d of %d" % (n,nev)

    chain.GetEntry(n)

    # Loop over the top and antitop truth info.
    for i in range(0,2):

        pt_truth = chain.GetLeaf(str_truth_pt).GetValue(i)
        eta_truth = chain.GetLeaf(str_truth_eta).GetValue(i)
        phi_truth = chain.GetLeaf(str_truth_phi).GetValue(i)
        mass_truth = chain.GetLeaf(str_truth_mass).GetValue(i)

        p4_truth.SetPtEtaPhiM(pt_truth,eta_truth,phi_truth,mass_truth);

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
            mass_meas = chain.GetLeaf(str_meas_mass).GetValue(j)

            # Make some pt cuts for now. May relax this. 
            # If reconstructed is within dR<0.8, this is a match!
            if pt_meas > 100 and pt_meas<1100:

                p4_meas.SetPtEtaPhiM(pt_meas,eta_meas,phi_meas,mass_meas);

                # Calc dR between truth and reconstructed jet.
                #dR = np.sqrt((eta_meas-eta_truth)**2 + (phi_meas-phi_truth)**2)

                dR0 = p4_meas.DeltaR(p4_truth);
                #print p4_truth.Pt(),p4_meas.Pt()
                #print "dR dR0: %f %f" % (dR,dR0)

                if dR0 < min_dR and dR0 < 0.8:
                    min_dR = dR0
                    min_dR_pt = pt_meas
                    found_match = True

        # Fill the response matrix accordingly.
        if found_match == False:
            response.Miss (pt_truth)
        else:
            hMC_meas.Fill(min_dR_pt)
            response.Fill (min_dR_pt, pt_truth)



################################################################################
# Unfold with one of the algorithms
################################################################################
unfold0 = RooUnfoldBayes(response,hMC_meas,4);
#unfold0 = RooUnfoldSvd(response,hMC_meas, 20);  


# MC true, measured, and unfolded histograms 
c1 = TCanvas( 'c1', 'MC', 200, 10, 700, 500 )

hMC_true.SetLineColor(kBlack);  
hMC_true.Draw();  # MC raw 
c1.SaveAs("MC_true.png")

hMC_meas.SetLineColor(kBlue);
hMC_meas.Draw("SAME");  # MC measured
c1.SaveAs("MC_meas.png")

hMC_reco = unfold0.Hreco();
hMC_reco.SetLineColor(kRed);
hMC_reco.Draw("SAME");        # MC unfolded 
c1.SaveAs("MC_unfold.png")

c1.Update()

# MC efficiency (meas/raw)
c2 = TCanvas( 'c2', 'MC_eff', 200, 10, 700, 500)

hMC_eff = hMC_meas.Clone();
hMC_eff.Divide(hMC_true);
c2.SetLogy();
hMC_eff.Draw();
c2.SaveAs("MC_eff.png")

c2.Update()


################################################################################
print "======================================Response matrix========================="
response.Mresponse().Print()

c5 = TCanvas('c5', 'Response matrix',200,10,1000,500)
c5.Divide(2,1)
c5.cd(1)
response.Hresponse().Draw("colz")
#response.Mresponse().Draw("colz")
c5.cd(2)
#response.Hresponse().Draw()
#response.Mresponse().Draw("")
response.Mresponse().Draw("colz")

################################################################################
print "======================================Correct the data========================="
hdata = TH1D("hdata","Data: Top quark p_{t}", 10,100,1100)
meas_data = [1000,500,100,100,50,45,35,34,20,10]
for i,d in enumerate(meas_data):
    hdata.SetBinContent(i,d)

unfold1 = RooUnfoldBayes(response,hdata,4);
hdata_reco = unfold1.Hreco();

c6 = TCanvas('c6', 'Corrected data',200,10,1000,500)
c6.Divide(2,1)
c6.cd(1)
hdata_reco.SetLineColor(kRed);
hdata.Draw();        # MC unfolded 
c6.cd(2)
hdata_reco.Draw();        # MC unfolded 
c6.SaveAs("data_unfold.png")

c6.Update()

################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

