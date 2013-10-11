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

ROOT.gStyle.SetOptStat(11)

# ==============================================================================
#  Example Unfolding
# ==============================================================================

# ==============================================================================
#  Read in the data.
# ==============================================================================
infile = open(sys.argv[1],"r")

ptlo = 100
pthi = 800
nptbins = 7

tag = "boosted_top"
        

print "==================================== TRAIN ===================================="
#response= RooUnfoldResponse (40,100,600)
#hMC_true = TH1D("MC_true","Truth and measured: Top quark p_{t}", 40, 100,600)
#hMC_meas = TH1D("MC_meas","Efficiency: Top quark p_{t}", 40, 100,600)

#response= RooUnfoldResponse (10,100,1100)
response= RooUnfoldResponse (nptbins,ptlo,pthi)
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
for truth,meas in zip(top_truth,top_meas):

    if n%10000==0:
        print "%d of %d" % (n,nentries/2)


    # Fill the response matrix accordingly.
    if truth>0 and meas<0:
        response.Miss(truth)
        hMC_true.Fill(truth)

    elif truth>0 and meas>0:
        hMC_true.Fill(truth)
        hMC_meas.Fill(meas)
        response.Fill(meas,truth)

    elif truth<0 and meas>0:
        hMC_meas.Fill(meas)
        response.Fake(meas)

    else:
        print "truth and meas issue!",truth,meas
        exit(-1)

    n+=1



################################################################################
# Unfold with one of the algorithms
################################################################################
unfold0 = RooUnfoldBayes(response,hMC_meas,4);
#unfold0 = RooUnfoldSvd(response,hMC_meas, 20);  


# MC true, measured, and unfolded histograms 
c1 = TCanvas( 'c1', 'MC', 200, 10, 700, 500 )

ROOT.gPad.SetLogy()

hMC_true.SetLineColor(kBlack);  
hMC_true.Draw();  # MC raw 
name = "MC_true_%s.png" % (tag)
c1.SaveAs(name)

hMC_meas.SetLineColor(kBlue);
hMC_meas.Draw("SAME");  # MC measured
name = "MC_meas_%s.png" % (tag)
c1.SaveAs(name)

hMC_reco = unfold0.Hreco();
hMC_reco.SetLineColor(kRed);
hMC_reco.Draw("SAME");        # MC unfolded 
name = "MC_unfold_%s.png" % (tag)
c1.SaveAs(name)

c1.Update()

# MC efficiency (meas/raw)
c2 = TCanvas( 'c2', 'MC_eff', 200, 10, 700, 500)

hMC_eff = hMC_meas.Clone();
hMC_eff.Divide(hMC_true);
c2.SetLogy();
hMC_eff.Draw();
name = "MC_eff_%s.png" % (tag)
c2.SaveAs(name)

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

name = "response_matrices_%s.png" % (tag)
c5.SaveAs(name)
################################################################################
print "======================================Get the old results========================="
xpts = np.array([26.2, 66.2, 126.2, 173.8, 228.8, 286.2, 356.2, 500, 600, 700, 800, 900])
ypts = np.array([0.003695, 0.006477, 0.005077, 0.002795, 0.001246, 0.000491, 0.000126, 0,0,0,0,0])
ypts *= 1000 # Because of how they're plotted.

xpts_err = np.array([30.0,40.0,50.0,60.0,60.0,80.0,50,50,50,50,50])
xpts_err /= 2.0
ypts_pcterr = np.array([4.3,4.0,3.8,5.1,4.6,6.5,9.0,1,1,1,1,1])
ypts_err = ypts*ypts_pcterr/100.0

old_results = ROOT.TGraphErrors(len(xpts),xpts,ypts,xpts_err,ypts_err)
old_results.SetLineWidth( 2 )
old_results.SetMarkerColor( 4 )
old_results.SetMarkerStyle( 21 )

print "======================================Correct the data========================="
hdata = TH1D("hdata","Data: Top quark p_{t}",nptbins,ptlo,pthi)
meas_data = [0,4,63,62,32,14,3]
for i,d in enumerate(meas_data):
    hdata.SetBinContent(i+1,d)

unfold1 = RooUnfoldBayes(response,hdata,4);
hdata_reco = unfold1.Hreco();

c6 = TCanvas('c6', 'Corrected data',200,10,1000,500)
c6.Divide(2,1)
c6.cd(1)
hdata.Draw("e");        # MC unfolded 

# Scale by luminosity
c6.cd(2)
lumi = 19.7e15
xsec_tot = 247.0e-12
bin_width = hdata_reco.GetBinCenter(2)-hdata_reco.GetBinCenter(1)
print "bin_width ",bin_width
hdata_reco.Scale(1.0/bin_width)
hdata_reco.SetLineColor(kRed);
hdata_reco.Scale(1000*(1.0/lumi)/xsec_tot)
hdata_reco.GetYaxis().SetTitle("#frac{1}{#sigma} #frac{d#sigma}{d p^{t}_{T}} [GeV^{-1}] #times 10^{-3}")
hdata_reco.GetXaxis().SetTitle("p^{t}_{T} [GeV^{-1}]")

ROOT.gPad.SetLogy()
old_results.GetYaxis().SetRangeUser(0.0001,8)
old_results.Draw("ap")
hdata_reco.Draw("samee");        # MC unfolded 
old_results.Draw("p")
name = "data_unfold_%s.png" % (tag)
c6.SaveAs(name)

c6.Update()

c7 = TCanvas('c7', 'Old data',520,50,1000,500)
c7.Divide(2,1)
c7.cd(1)
old_results.Draw("ap")
c7.Update()

################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

