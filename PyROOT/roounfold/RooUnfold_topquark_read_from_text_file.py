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

infile = open(sys.argv[1],"r")

ptlo = 100
pthi = 800
nptbins = 7
        

print "==================================== TRAIN ===================================="
#response= RooUnfoldResponse (40,100,600)
#hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 40, 100,600)
#hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 40, 100,600)

response= RooUnfoldResponse (10,100,1100)
hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}",nptbins,ptlo,pthi)
hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}",nptbins,ptlo,pthi)


################################################################################
# Loop over the events
################################################################################
vals = (np.array(infile.read().split())).astype(float)
nentries = len(vals)
ncols = 2
index = np.arange(0,nentries,2)
top_truth = vals[index]
top_meas = vals[index+1]

n = 0
for tt,tr in zip(top_truth,top_meas):

    if n%10000==0:
        print "%d of %d" % (n,nentries/2)

    hMC_true.Fill(tt)

    # Fill the response matrix accordingly.
    if tr<0:
        #response.Miss(tt)
        response.Fake(tt)
    else:
        hMC_meas.Fill(tr)
        response.Fill(tr,tt)

    n+=1



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
hdata = TH1D("hdata","Data: Top quark p_{t}",nptbins,ptlo,pthi)
#meas_data = [1000,500,100,100,50,45,35,34,20,10]
meas_data = [0,0,15,30,20,10,5]
for i,d in enumerate(meas_data):
    hdata.SetBinContent(i,d)

unfold1 = RooUnfoldBayes(response,hdata,4);
hdata_reco = unfold1.Hreco();

c6 = TCanvas('c6', 'Corrected data',200,10,1000,500)
c6.Divide(2,1)
c6.cd(1)
hdata_reco.SetLineColor(kRed);

# Scale by luminosity
lumi = 19.7e15
xsec_tot = 247.0e-12
hdata_reco.Scale((1.0/lumi)/xsec_tot)
hdata.Draw();        # MC unfolded 

c6.cd(2)
ROOT.gPad.SetLogy()
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

