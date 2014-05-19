#!/bin/tcsh -f 

source kipac.csh

#foreach baryon ( "LambdaC" "Lambda0" )
    #foreach ntp ( "ntp1" "ntp2" "ntp3" "ntp4" )

foreach baryon ( "LambdaC" )
#foreach baryon ( "Lambda0" )
    #foreach ntp ( "ntp1" "ntp2" "ntp3" "ntp4" )
    foreach ntp ( "ntp1" "ntp2" )

        echo python plot_fit_results.py \
               --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_bkg_pass0_nfits1.root \
               --fit-only-bkg \
               --num-bins 50 \
               --batch

        python plot_fit_results.py \
               --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_bkg_pass0_nfits1.root \
               --fit-only-bkg \
               --num-bins 50 \
               --batch

        python plot_fit_results.py \
               --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_sig_pass0_nfits1.root \
               --fit-only-sig \
               --num-bins 100 \
               --batch

    end
end
