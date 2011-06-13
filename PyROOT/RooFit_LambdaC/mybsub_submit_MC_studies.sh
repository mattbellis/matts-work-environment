#!/bin/tcsh 

set baryon = $1
set task = $2 # fit, extract, summarize
set no_gc = $3

foreach ntp ( 1 2 3 4)
    if ( ! ( $baryon == "LambdaC" && ( $ntp == 3 || $ntp == 4 ) ) ) then
    foreach p_or_e ( "pure" "embed" )
        #foreach num (0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 50 60 70 80 90 100)
        foreach num (0 3 6 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 5 30)
            #foreach dim( "3" "2" )
            foreach dim( "2" )
            if ( ! (  $baryon == "Lambda0" && $dim == '3' ) ) then
                foreach fixnum( ' ' ' --fixed-num ' )

                    echo $task
                    if ( $task == 'fit' ) then
                        set logfile = "./batch_logs/log_`date +%s`_toystudies_ntp"$ntp"_num"$num"_"$p_or_e".log" 
                        bsub -R rhel50 -q long -o $logfile "./bcmd_run_MC_studies.sh $baryon $ntp $num --$p_or_e --dim $dim $fixnum $no_gc"
                    else if ( $task == 'extract' ) then
                        set logfile = "./batch_logs/log_"$baryon"_extract_`date +%s`_toystudies_ntp"$ntp"_num"$num"_"$p_or_e".log"
                        bsub -R rhel50 -q long -o $logfile "./bcmd_extract_MC_studies.sh $baryon $ntp $num --$p_or_e --dim $dim $fixnum $no_gc"
                    else if ( $task == 'summarize' ) then
                        set logfile = "./batch_logs/"$baryon"_summarize_log_`date +%s`_toystudies_ntp"$ntp"_num"$num"_"$p_or_e".log"
                        bsub -R rhel50 -q long -o $logfile "./bcmd_summarize_MC_studies.sh $baryon $ntp $num --$p_or_e --dim $dim $fixnum $no_gc"
                    endif

                end
            else
                echo "Not submitting for $baryon $dim"
            endif 
            end
        end
    end
    else
        echo "Not submitting for $baryon $ntp"
    endif
end


