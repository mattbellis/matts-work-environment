#!/bin/tcsh 

source kipac.csh

foreach baryon ( "LambdaC" )
    #foreach ntp ( "ntp1" "ntp2" )
    foreach ntp ( "ntp2" )

    set tag = $baryon"_"$ntp"_forpaper"

    python ./read_in_and_fit.py -d 3 \
                                --baryon "$baryon" \
                                --ntp "$ntp" \
                                --num-sig 0 \
                                --num-bkg 500 \
                                --starting-vals startingValuesForFits/values_for_fits_"$baryon"_"$ntp"_pass0.txt \
                                --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_sig_pass0_nfits1.root \
                                mcstudy_"$baryon"_"$ntp"_pass0/mcstudies_bkg*_sig0_0101.dat \
                                --tag $tag \
                                --pass 0 \
                                --fit \
                                --sideband-first \
                                --batch

    python ./plot_fit_results.py --workspace rootWorkspaceFiles/workspace_"$tag"_sig0_bkg500_dim3_nfits1.root --num-bins 50 --batch --tag $tag

    end
end


exit(0)



foreach baryon ( "Lambda0" )
    foreach ntp ( "ntp1" "ntp2" "ntp3" "ntp4" )

    set tag = $baryon"_"$ntp"_forpaper"

    python ./read_in_and_fit.py -d 2 \
                                --baryon "$baryon" \
                                --ntp "$ntp" \
                                --num-sig 0 \
                                --num-bkg 500 \
                                --starting-vals startingValuesForFits/values_for_fits_"$baryon"_"$ntp"_pass0.txt \
                                --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_sig_pass0_nfits1.root \
                                mcstudy_"$baryon"_"$ntp"_pass0/mcstudies_bkg*_sig0_0100.dat \
                                --tag $tag \
                                --pass 0 \
                                --fit \
                                --sideband-first \
                                --batch

    python ./plot_fit_results.py --workspace rootWorkspaceFiles/workspace_"$tag"_sig0_bkg500_dim2_nfits1.root --num-bins 50 --batch --tag $tag

    end
end


#python ./plot_fit_results.py --workspace rootWorkspaceFiles/workspace_LambdaC_ntp2_forpaper_sig0_bkg500_dim3_nfits1.root --num-bins 50 --batch --tag LambdaC_ntp2_forpaper

