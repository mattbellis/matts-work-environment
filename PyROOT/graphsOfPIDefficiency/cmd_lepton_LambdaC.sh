#############################################
# LambdaC mu/e
#############################################
# mu
#./plotTheIndividualHistograms_lepton.py mu_v03_ntp1_sigbak_sp9446.root  LambdaC_mu_SP9446      xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py mu_v03_ntp1_sigbak_onpeak.root  LambdaC_mu_onpeak      xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py allSP_v03_mu.root               LambdaC_mu_genericMC   xxxxxxxx batch
# e
#./plotTheIndividualHistograms_lepton.py lepton_v03_ntp3_sigbak_sp9445.root  LambdaC_e_SP9445      xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py lepton_v03_ntp3_sigbak_onpeak.root  LambdaC_e_onpeak      xxxxxxxx batch
#./plotTheIndividualHistograms_lepton.py leptonLambdaC_combinedGenerics_e.root LambdaC_e_genericMC   xxxxxxxx batch

#exit()

set bkg = "genericMC"
#set bkg = "onpeak"
set lep = "e"

set sigsp = "9446"
if ( $lep == "e" ) then
  set sigsp = "9445"
endif

# S/sqrt(B0)
./calc_best_lept_selector.py "$lep" outVals_LambdaC_"$lep"_$bkg.txt outVals_LambdaC_"$lep"_SP"$sigsp".txt 10 200 0
# Punzi
./calc_best_lept_selector.py "$lep" outVals_LambdaC_"$lep"_$bkg.txt outVals_LambdaC_"$lep"_SP"$sigsp".txt 10 20  2
./calc_best_lept_selector.py "$lep" outVals_LambdaC_"$lep"_$bkg.txt outVals_LambdaC_"$lep"_SP"$sigsp".txt 10 50  2
./calc_best_lept_selector.py "$lep" outVals_LambdaC_"$lep"_$bkg.txt outVals_LambdaC_"$lep"_SP"$sigsp".txt 10 100 2
./calc_best_lept_selector.py "$lep" outVals_LambdaC_"$lep"_$bkg.txt outVals_LambdaC_"$lep"_SP"$sigsp".txt 10 200 2



