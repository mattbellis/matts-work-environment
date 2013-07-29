#!/usr/bin/env python
# ==============================================================================
#  File and Version Information:
#       $Id: RooUnfoldExample.py 302 2011-09-30 20:39:20Z T.J.Adye $
#
#  Description:
#       Simple example usage of the RooUnfold package using toy MC.
#
#  Author: Tim Adye <T.J.Adye@rl.ac.uk>
#
# ==============================================================================

from ROOT import gRandom, TH1, TH1D, cout, TCanvas
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
response= RooUnfoldResponse (40,100,600)
hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 40, 100,600)
hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 40, 100,600)

# Truth?
str_truth_pt = "floats_pfShyftTupleTopQuarks_pt_ANA.obj"
str_truth_eta = "floats_pfShyftTupleTopQuarks_eta_ANA.obj"
str_truth_phi = "floats_pfShyftTupleTopQuarks_phi_ANA.obj"

# Reconstructed
str_meas_pt = "floats_pfShyftTupleJetsLooseTopTag_pt_ANA.obj"
str_meas_eta = "floats_pfShyftTupleJetsLooseTopTag_eta_ANA.obj"
str_meas_phi = "floats_pfShyftTupleJetsLooseTopTag_phi_ANA.obj"

chain.SetBranchStatus('*', 1 )
#chain.SetBranchStatus(truth, 1 )
#chain.SetBranchStatus(meas, 1 )

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

for n in xrange(nev):
    if n%1000==0:
        print n

    chain.GetEntry(n)

    for i in range(0,2):

        pt_truth = chain.GetLeaf(str_truth_pt).GetValue(i)
        eta_truth = chain.GetLeaf(str_truth_eta).GetValue(i)
        phi_truth = chain.GetLeaf(str_truth_phi).GetValue(i)

        hMC_true.Fill(pt_truth)

        found_match = False
        min_dR = 100000.0
        min_dR_pt = -1
        min_index = -1

        for j in range(0,2):

            pt_meas = chain.GetLeaf(str_meas_pt).GetValue(j)
            eta_meas = chain.GetLeaf(str_meas_eta).GetValue(j)
            phi_meas = chain.GetLeaf(str_meas_phi).GetValue(j)

            dR = np.sqrt((eta_meas-eta_truth)**2 + (phi_meas-phi_truth)**2)

            if pt_meas > 100 and pt_meas<600:
                if dR < min_dR and dR < 0.8:
                    min_dR = dR
                    min_dR_pt = pt_meas
                    found_match = True

        if found_match == False:
            response.Miss (pt_truth)
        else:
            hMC_meas.Fill(min_dR_pt)
            response.Fill (min_dR_pt, pt_truth)



    '''
    if val_meas>100 and val_meas<600:
        hMC_meas.Fill(val_meas)

    if val_meas>100 and val_meas<600:
        response.Fill (val_meas, val_truth)
    else:
        response.Miss (val_truth)
    '''

 

'''
#  Train with a Breit-Wigner, mean 0.3 and width 2.5.
for i in xrange(100000):
  xt= gRandom.BreitWigner (0.3, 2.5);
  x= smear (xt);
  hMC_true.Fill(xt);
  if x!=None:
    response.Fill (x, xt);
    hMC_meas.Fill(x);
  else:
    response.Miss (xt);

'''

#unfold0 = RooUnfoldBayes(response,hMC_meas,4);
unfold0 = RooUnfoldSvd(response,hMC_meas, 20);  


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

'''
print "==================================== TEST ====================================="
hTrue= TH1D ("true", "Test Measured: Gaussian",    40, -10.0, 10.0);
hMeas= TH1D ("meas", "Test Efficiency: Gaussian", 40, -10.0, 10.0);
#  Test with a Gaussian, mean 0 and width 2.
for i in xrange(10000):
  xt= gRandom.Gaus (0.0, 2.0)
  x= smear (xt);
  hTrue.Fill(xt);
  if x!=None:
    hMeas.Fill(x);

# Data efficiency (meas/raw)
c4 = TCanvas( 'c4', 'Data_eff', 200, 10, 700, 500)

hData_eff = hMeas.Clone();
hData_eff.Divide(hTrue);
c4.SetLogy();
hData_eff.Draw();
c4.SaveAs("Data_eff.png")

c4.Update()


print "==================================== UNFOLD ==================================="
# unfold= RooUnfoldBayes     (response, hMeas, 4);    #  OR
unfold= RooUnfoldSvd     (response, hMeas, 20);   #  OR
# unfold= RooUnfoldTUnfold (response, hMeas);


# Data true, measured and unfolded histograms 
c3 = TCanvas( 'c3', 'Data', 200, 10, 700, 500 )

hTrue.SetLineColor(kBlack);
hTrue.Draw();     # Data raw
c3.SaveAs("Data_true.png")

hMeas.SetLineColor(kBlue);
hMeas.Draw("SAME");     # Data measured
c3.SaveAs("Data_meas.png")

hReco= unfold.Hreco();
unfold.PrintTable (cout, hTrue);
hReco.SetLineColor(kRed);
hReco.Draw("SAME");           # Data unfolded 
c3.SaveAs("Data_unfold.png")

c3.Update()
'''

#================================================================================
print "======================================Response matrix========================="
response.Mresponse().Print()

c5 = TCanvas('c5', 'Response matrix',200,10,1000,500)
c5.Divide(2,1)
c5.cd(1)
#response.Hresponse().Draw()
response.Mresponse().Draw("colz")
c5.cd(2)
#response.Hresponse().Draw()
response.Mresponse().Draw("")


################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

