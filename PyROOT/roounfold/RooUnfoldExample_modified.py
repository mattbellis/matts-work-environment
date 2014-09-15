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
#from ROOT import RooUnfoldBayes
from ROOT import kRed,kBlack,kBlue
from ROOT import RooUnfoldSvd
#from ROOT import RooUnfoldTUnfold

# ==============================================================================
#  Gaussian smearing, systematic translation, and variable inefficiency
# ==============================================================================

def smear(xt):
  xeff= 0.3 + (1.0-0.3)/20*(xt+10.0);  #  efficiency
  x= gRandom.Rndm();
  if x>xeff: return None;
  xsmear= gRandom.Gaus(-2.5,0.2);     #  bias and smear
  return xt+xsmear;

# ==============================================================================
#  Example Unfolding
# ==============================================================================

print "==================================== TRAIN ===================================="
response= RooUnfoldResponse (40, -10.0, 10.0);
hMC_true = TH1D("MC_true","Truth Measured: Breit-Wigner", 40, -10.0, 10);
hMC_meas = TH1D("MC_meas","Truth Efficiency: Breit-Wigner", 40, -10.0, 10);

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

################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

