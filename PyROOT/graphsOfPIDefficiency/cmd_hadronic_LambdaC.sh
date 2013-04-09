#!/bin/tcsh

set ntp = $1
#set lepsel = $2

set spsig = 9446
set lepsel = 1
if ( $ntp == "ntp1" ) then
  set spsig = 9446
  set lepsel = 6
else if ( $ntp == "ntp2" ) then
  set spsig = 9445
  set lepsel = 1
endif

set dataouttag = LambdaC_onpeak_"$ntp"_lep"$lepsel" 
#./plotTheIndividualHistograms.py LambdaCLambdaC_lep"$lepsel"_"$ntp"_sigbak_onpeak.root $dataouttag xxxxxxxx batch 

set sigouttag = LambdaC_sig_"$ntp"_lep"$lepsel" 
#./plotTheIndividualHistograms.py LambdaCLambdaC_lep"$lepsel"_"$ntp"_sigbak_sp$spsig.root $sigouttag xxxxxxxx batch 

set genericsouttag = LambdaC_generics_"$ntp"_lep"$lepsel" 
#./plotTheIndividualHistograms.py LambdaCLambdaC_lep"$lepsel"_"$ntp"_combinedGenerics.root $genericsouttag xxxxxxxx batch 




#./plotTwoDifferentFits.py Signaleff               Backgroundeff           0 0 sig $lepsel LambdaC_OnOn_"$ntp"_lep"$lepsel" nobatch
#./plotTwoDifferentFits.py outVals_$dataouttag.txt outVals_$dataouttag.txt 0 0 sig $lepsel LambdaC_OnOn_"$ntp"_lep"$lepsel" nobatch
./plotTwoDifferentFits.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 1   nosig   $lepsel LambdaC_SigGenerics_"$ntp"_lep"$lepsel"_soversqrtB nobatch
./plotTwoDifferentFits.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 20  sig     $lepsel LambdaC_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg20 nobatch
./plotTwoDifferentFits.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 50  sig     $lepsel LambdaC_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg50 nobatch
./plotTwoDifferentFits.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 100 sig     $lepsel LambdaC_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg100 nobatch
./plotTwoDifferentFits.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 200 sig     $lepsel LambdaC_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg200 nobatch

#./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 1    nosig Lambda0_SigGenerics_"$ntp"_lep"$lepsel"_soversqrtB  nobatch
#./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 20   sig   Lambda0_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg20 nobatch
#./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 50   sig   Lambda0_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg50 nobatch
