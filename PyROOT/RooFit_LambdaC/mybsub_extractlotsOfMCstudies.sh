#!/bin/tcsh 

foreach num(0 10 20 30 40 50 60 70 80 90 100)
#foreach num( 10 20 30 40 50 60 70 80 90 100)
    bsub -R rhel50 -q long -o ./batch_logs/log_`date +%s`_toystudies_extract_num$num.log "bcmd_LambdaC_extract_MC_studies.sh $num"
end
