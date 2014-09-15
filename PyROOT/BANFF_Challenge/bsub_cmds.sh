#!/bin/tcsh -f

set nsig = $1
@ nbootstrap_samples = $2
set distance = $3
set lo = $4
set hi = $5

set queue = 'medium'

if ( $nbootstrap_samples > 100 ) then
    set queue = 'long'
endif


@ i = 0


set logfile = "logfiles/log_nsig$nsig"_"nbs$nbootstrap_samples"_r"$distance"_$lo"_"$hi.dat
echo $logfile

bsub -q $queue -o $logfile shell_script_for_batch_running_multiple_jobs.sh $nsig $nbootstrap_samples $distance $lo $hi 

@ i++

