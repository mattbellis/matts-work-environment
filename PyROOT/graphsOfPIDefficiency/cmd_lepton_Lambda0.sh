#############################################
# Lambda0 mu/e
#############################################

set ntp = $1
#set bkg = "combinedGenerics"
set bkg = "onpeak"

set sigsp = "9446"
set lep = "e"
if ( $ntp == "ntp1" ) then
  set sigsp = "9454"
  set lep = "mu"
else if ( $ntp == "ntp2" ) then
  set sigsp = "9455"
  set lep = "e"
else if ( $ntp == "ntp3" ) then
  set sigsp = "9452"
  set lep = "mu"
else if ( $ntp == "ntp4" ) then
  set sigsp = "9453"
  set lep = "e"
endif


#./plotTheIndividualHistograms_lepton.py $lep leptonLambda0_combinedGenerics_"$ntp".root       Lambda0_lep_"$ntp"_combinedGenerics xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py $lep leptonLambda0_lep0_"$ntp"_sigbak_onpeak.root     Lambda0_lep_"$ntp"_onpeak           xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py $lep leptonLambda0_lep0_"$ntp"_sigbak_sp"$sigsp".root Lambda0_lep_"$ntp"_sp"$sigsp"       xxxxxxxx batch

#exit()



# S/sqrt(B0)
./calc_best_lept_selector.py "$lep" outVals_Lambda0_lep_"$ntp"_$bkg.txt outVals_Lambda0_lep_"$ntp"_sp"$sigsp".txt 10 200 0
# Punzi
./calc_best_lept_selector.py "$lep" outVals_Lambda0_lep_"$ntp"_$bkg.txt outVals_Lambda0_lep_"$ntp"_sp"$sigsp".txt 10 20  2
./calc_best_lept_selector.py "$lep" outVals_Lambda0_lep_"$ntp"_$bkg.txt outVals_Lambda0_lep_"$ntp"_sp"$sigsp".txt 10 50  2
./calc_best_lept_selector.py "$lep" outVals_Lambda0_lep_"$ntp"_$bkg.txt outVals_Lambda0_lep_"$ntp"_sp"$sigsp".txt 10 100 2
./calc_best_lept_selector.py "$lep" outVals_Lambda0_lep_"$ntp"_$bkg.txt outVals_Lambda0_lep_"$ntp"_sp"$sigsp".txt 10 200 2



