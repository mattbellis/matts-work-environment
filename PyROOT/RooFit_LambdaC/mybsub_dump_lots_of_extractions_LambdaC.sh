#!/bin/tcsh 

foreach ntp ( 1 2 )
    foreach p_or_e ( "--pure" "--embed" )
        foreach fixed_or_not ( "--fixed-num" " " )
            foreach num(0 10 20 30 40 50 60 70 80 90 100    2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36)
                foreach dim( ' --dim 2' )
                  set logfile = "./batch_logs/log_LambdaC_extract_`date +%s`_toystudies_ntp"$ntp"_num"$num"_"$p_or_e".log"
                  bsub -R rhel50 -q long -o $logfile "./bcmd_LambdaC_extract_MC_studies.sh $ntp $num $p_or_e $fixed_or_not $dim"
                end
            end
        end
    end
end


