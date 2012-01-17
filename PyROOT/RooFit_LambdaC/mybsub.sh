#!/bin/tcsh 
bsub -q long -o ./batch_logs/log_`date +%s`_$1.log "$1 $2"
