#!/bin/tcsh

set ntp = $1
#set lepsel = $2

set spsig = 9446
set lepsel = 1
if ( $ntp == "ntp1" ) then
  set spsig = 9454
  set lepsel = 6
else if ( $ntp == "ntp2" ) then
  set spsig = 9455
  set lepsel = 3
else if ( $ntp == "ntp3" ) then
  set spsig = 9452
  set lepsel = 6
else if ( $ntp == "ntp4" ) then
  set spsig = 9453
  set lepsel = 3
endif

set dataouttag = Lambda0_hs_onpeak_"$ntp"_lep"$lepsel" 
./plotTheIndividualHistograms_Lambda0.py Lambda0Lambda0_lep"$lepsel"_"$ntp"_sigbak_onpeak.root $dataouttag xxxxxxxx batch

set sigouttag = Lambda0_hs_sig_"$ntp"_lep"$lepsel" 
./plotTheIndividualHistograms_Lambda0.py Lambda0Lambda0_lep"$lepsel"_"$ntp"_sigbak_sp$spsig.root $sigouttag xxxxxxxx batch

set genericsouttag = Lambda0_hs_generics_"$ntp"_lep"$lepsel" 
./plotTheIndividualHistograms_Lambda0.py Lambda0Lambda0_lep"$lepsel"_"$ntp"_combinedGenerics.root $genericsouttag xxxxxxxx batch




#./plotTwoDifferentFits_Lambda0.py Signaleff               Backgroundeff           0 0 sig    Lambda0_OnOn_"$ntp"_lep"$lepsel" batch
#./plotTwoDifferentFits_Lambda0.py outVals_$dataouttag.txt outVals_$dataouttag.txt 0 0 sig    Lambda0_OnOn_"$ntp"_lep"$lepsel" batch

./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 1    nosig Lambda0_hs_SigGenerics_"$ntp"_lep"$lepsel"_soversqrtB  batch
./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 20   sig   Lambda0_hs_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg20 batch
./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 50   sig   Lambda0_hs_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg50 batch
./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 100  sig   Lambda0_hs_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg100 batch
./plotTwoDifferentFits_Lambda0.py outVals_$sigouttag.txt outVals_$genericsouttag.txt 1 200  sig   Lambda0_hs_SigGenerics_"$ntp"_lep"$lepsel"_punzi_bkg200 batch
