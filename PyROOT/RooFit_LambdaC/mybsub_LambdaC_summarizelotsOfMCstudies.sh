#!/bin/tcsh 

foreach ntp ( 1 2 )
    foreach p_or_e ( "pure" "embed" )
        foreach num(0 10 20 30 40 50 60 70 80 90 100)
            foreach dim( ' --dim 2' )
                set logfile = "./batch_logs/LambdaC_summarize_log_`date +%s`_toystudies_ntp"$ntp"_num"$num"_"$p_or_e".log" 
                bsub -R rhel50 -q long -o $logfile "./bcmd_LambdaC_summarize_MC_studies.sh $ntp $num --$p_or_e $dim"
            end
        end
    end
end

foreach ntp ( 1 2 )
    foreach p_or_e ( "pure" "embed" )
        foreach num (0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36)
            foreach dim( ' --dim 2' )
                set logfile = "./batch_logs/LambdaC_summarize_log_`date +%s`_toystudies_ntp"$ntp"_num"$num"_fixed_"$p_or_e".log" 
                bsub -R rhel50 -q long -o $logfile "./bcmd_LambdaC_summarize_MC_studies.sh $ntp $num --$p_or_e --fixed-num $dim"
            end
        end
    end
end

