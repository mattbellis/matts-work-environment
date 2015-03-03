#correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_fitRegion_fitVariables_onlyLikelihood.root 99 AllSP_signalRegion batch
#correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_fitRegion_fitVariables_onlyLikelihood.root               99 qqbarSP_signalRegion batch

#correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_fullEnergyRegion_fitVariables_onlyLikelihood.root 99 AllSP_fullEnergyRange batch
#correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_fullEnergyRegion_fitVariables_onlyLikelihood.root               99 qqbarSP_fullEnergyRange batch

foreach cut( 0 1 2 3 4 5 6 )
  correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_Ecut"$cut"_fitVariables_onlyLikelihood.root               99 AllSP_Ecut$cut batch
  correlations.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut"$cut"_fitVariables_onlyLikelihood.root 99 qqbarSP_Ecut$cut batch
end
./correlationscatters.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_Ecut0_fitVariables_onlyLikelihood.root BpostFitMes 0
./correlationscatters.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_Ecut0_fitVariables_onlyLikelihood.root BpostFitDeltaE AllSP_Ecut0
./correlationscatters.py rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_Ecut1_fitVariables_onlyLikelihood.root BpostFitDeltaE AllSP_Ecut1


./mygeteffs.py test rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_useAllForTraining.root rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_split*of10.root

mygeteffs.py test rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_split*of2.root rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_split*of5.root rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_split*of10.root


mygeteffs.py crossvars10_nbkg16_wbest rootFiles/TMVA_LambdaC_SP9446_SP1005_SP998_SP1237_SP1235_Ecut5_noBThrust_noCones_split*of10.root


##########################
# Latest command
##########################
./mygeteffs.py rootFiles/*Lambda0*9454*5_E*of10*root rootFiles/*Lambda0*9454*5_E*of5*root rootFiles/*Lambda0*9454*5_E*of2*root

