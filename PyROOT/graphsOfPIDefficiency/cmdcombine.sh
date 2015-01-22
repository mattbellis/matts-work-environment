#!/bin/tcsh -f

#./combineFilesOfHistos.py allSP_v03_LambdaC.root \
                          #LambdaC_v03_ntp1_sigbak_sp1005.root 0.5 \
                          #LambdaC_v03_ntp1_sigbak_sp998.root 0.5 \
                          #LambdaC_v03_ntp1_sigbak_sp1235.root 0.35 \
                          #LambdaC_v03_ntp1_sigbak_sp1237.root 0.35 

#./combineFilesOfHistos.py allSP_v03_mu.root \
                          #mu_v03_ntp1_sigbak_sp1005.root 0.5 \
                          #mu_v03_ntp1_sigbak_sp998.root 0.5 \
                          #mu_v03_ntp1_sigbak_sp1235.root 0.35 \
                          #mu_v03_ntp1_sigbak_sp1237.root 0.35 

#./combineFilesOfHistos.py leptonLambdaC_combinedGenerics_e.root \
                          #leptonLambdaC_lep0_ntp2_sigbak_sp1005.root 0.5 \
                          #leptonLambdaC_lep0_ntp2_sigbak_sp998.root 0.5 \
                          #leptonLambdaC_lep0_ntp2_sigbak_sp1235.root 0.35 \
                          #leptonLambdaC_lep0_ntp2_sigbak_sp1237.root 0.35 

#foreach ntp ( ntp1 ntp2 ntp3 ntp4)
#foreach ntp ( ntp1 ntp3 )

#./combineFilesOfHistos.py leptonLambda0_combinedGenerics_"$ntp".root \
  #leptonLambda0_lep0_"$ntp"_sigbak_sp1005.root 0.5 \
    #leptonLambda0_lep0_"$ntp"_sigbak_sp998.root 0.5 \
      #leptonLambda0_lep0_"$ntp"_sigbak_sp1235.root 0.35 \
        #leptonLambda0_lep0_"$ntp"_sigbak_sp1237.root 0.35 
        #
        #end


        #./combineFilesOfHistos.py LambdaCLambdaC_lep6_ntp1_combinedGenerics.root \
          #LambdaCLambdaC_lep6_ntp1_sigbak_sp1005.root 0.50 \
            #LambdaCLambdaC_lep6_ntp1_sigbak_sp998.root  0.50 \
              #LambdaCLambdaC_lep6_ntp1_sigbak_sp1235.root 0.35 \
                #LambdaCLambdaC_lep6_ntp1_sigbak_sp1237.root 0.35

                #./combineFilesOfHistos.py LambdaCLambdaC_lep1_ntp2_combinedGenerics.root \
                  #LambdaCLambdaC_lep1_ntp2_sigbak_sp1005.root 0.50 \
                    #LambdaCLambdaC_lep1_ntp2_sigbak_sp998.root  0.50 \
                      #LambdaCLambdaC_lep1_ntp2_sigbak_sp1235.root 0.35 \
                        #LambdaCLambdaC_lep1_ntp2_sigbak_sp1237.root 0.35

./combineFilesOfHistos.py Lambda0Lambda0_lep6_ntp1_combinedGenerics.root \
                          Lambda0Lambda0_lep6_ntp1_sigbak_sp1005.root 0.50 \
                          Lambda0Lambda0_lep6_ntp1_sigbak_sp998.root  0.50 \
                          Lambda0Lambda0_lep6_ntp1_sigbak_sp1235.root 0.35 \
                          Lambda0Lambda0_lep6_ntp1_sigbak_sp1237.root 0.35

./combineFilesOfHistos.py Lambda0Lambda0_lep3_ntp2_combinedGenerics.root \
                          Lambda0Lambda0_lep3_ntp2_sigbak_sp1005.root 0.50 \
                          Lambda0Lambda0_lep3_ntp2_sigbak_sp998.root  0.50 \
                          Lambda0Lambda0_lep3_ntp2_sigbak_sp1235.root 0.35 \
                          Lambda0Lambda0_lep3_ntp2_sigbak_sp1237.root 0.35

./combineFilesOfHistos.py Lambda0Lambda0_lep6_ntp3_combinedGenerics.root \
                          Lambda0Lambda0_lep6_ntp3_sigbak_sp1005.root 0.50 \
                          Lambda0Lambda0_lep6_ntp3_sigbak_sp998.root  0.50 \
                          Lambda0Lambda0_lep6_ntp3_sigbak_sp1235.root 0.35 \
                          Lambda0Lambda0_lep6_ntp3_sigbak_sp1237.root 0.35

./combineFilesOfHistos.py Lambda0Lambda0_lep3_ntp4_combinedGenerics.root \
                          Lambda0Lambda0_lep3_ntp4_sigbak_sp1005.root 0.50 \
                          Lambda0Lambda0_lep3_ntp4_sigbak_sp998.root  0.50 \
                          Lambda0Lambda0_lep3_ntp4_sigbak_sp1235.root 0.35 \
                          Lambda0Lambda0_lep3_ntp4_sigbak_sp1237.root 0.35



