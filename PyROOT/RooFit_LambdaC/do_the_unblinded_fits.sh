#!/bin/tcsh 

source kipac.csh

echo

set baryons = ( "LambdaC" "LambdaC" "Lambda0" "Lambda0" "Lambda0" "Lambda0" )
set ntps = ( "ntp1" "ntp2" "ntp1" "ntp2" "ntp3" "ntp4" )

set input_files = ( "mcstudy_LambdaC_ntp1_pass0/mcstudies_bkg800_sig0_0101.dat" \
                    "mcstudy_LambdaC_ntp2_pass0/mcstudies_bkg620_sig0_0101.dat" \
                    "mcstudy_Lambda0_ntp1_pass0/mcstudies_bkg450_sig0_0101.dat" \
                    "mcstudy_Lambda0_ntp2_pass0/mcstudies_bkg430_sig0_0101.dat" \
                    "mcstudy_Lambda0_ntp3_pass0/mcstudies_bkg370_sig0_0101.dat" \
                    "mcstudy_Lambda0_ntp4_pass0/mcstudies_bkg120_sig0_0101.dat" \
                    )

@ i = 1

while ( $i < 7 )

    set baryon =  $baryons[$i]
    set ntp =  $ntps[$i]
    set input_file =  $input_files[$i]

    foreach gc ( "" "--no-gc" )

        set tag = $baryon"_"$ntp"_unblind"
        
        #set cmd = "python ./read_in_and_fit.py"
        set cmd = "./read_in_and_fit.py"
        set cmd = "$cmd -d 3"
        set cmd = "$cmd --baryon $baryon" 
        set cmd = "$cmd --ntp $ntp" 
        set cmd = "$cmd --num-sig 0"
        set cmd = "$cmd --num-bkg 500" 
        set cmd = "$cmd --starting-vals startingValuesForFits/values_for_fits_"$baryon"_"$ntp"_pass0.txt"
        set cmd = "$cmd --workspace rootWorkspaceFiles/workspace_determinedValues_"$baryon"_"$ntp"_sig_pass0_nfits1.root"
        set cmd = "$cmd --tag $tag"
        set cmd = "$cmd --no-plots"
        set cmd = "$cmd $gc"
        set cmd = "$cmd --pass 0"
        set cmd = "$cmd --fit"
        set cmd = "$cmd --sideband-first"
        set cmd = "$cmd --batch"
        set cmd = "$cmd $input_file"

        echo python $cmd 
        python $cmd

        #python ./plot_fit_results.py --workspace rootWorkspaceFiles/workspace_"$tag"_sig0_bkg500_dim3_nfits1.root --num-bins 50 --batch --tag $tag

        @ i++

        echo

    end
end


exit(0)


