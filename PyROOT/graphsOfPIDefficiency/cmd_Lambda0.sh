#!/bin/tcsh 

foreach ntp( 1 2 3 4 )
  ./plotTheIndividualHistograms_Lambda0.py Lambda0_v03_ntp"$ntp"_sigbak_OnPeak.root Lambda0_OnPeak_ntp$ntp      xxxxxxxx  batch
end

#./plotTwoDifferentFits_Lambda0.py outVals_Lambda0_OnPeak_ntp3.txt outVals_Lambda0_OnPeak_ntp3.txt 1 1   nosig OnPeakLambda0ntp3_nosig
#./plotTwoDifferentFits_Lambda0.py outVals_Lambda0_OnPeak_ntp3.txt outVals_Lambda0_OnPeak_ntp3.txt 10 1  sig   OnPeakLambda0ntp3_sig10_bkg1
#./plotTwoDifferentFits_Lambda0.py outVals_Lambda0_OnPeak_ntp3.txt outVals_Lambda0_OnPeak_ntp3.txt 100 1 sig   OnPeakLambda0ntp3_sig100_bkg1
#./plotTwoDifferentFits_Lambda0.py outVals_Lambda0_OnPeak_ntp3.txt outVals_Lambda0_OnPeak_ntp3.txt 1 10  sig   OnPeakLambda0ntp3_sig1_bkg10
#./plotTwoDifferentFits_Lambda0.py outVals_Lambda0_OnPeak_ntp3.txt outVals_Lambda0_OnPeak_ntp3.txt 1 100 sig   OnPeakLambda0ntp3_sig1_bkg100
