#!/usr/bin/env tcsh 

set fit = "2"
set fitname = "SHM_fixed_sigma_n_free_DM"

set logdir = "logs_"$fitname

if ( ! -e $logdir ) then
    mkdir $logdir
endif

ls $logdir

foreach base ( 39 40 41 42 )
    foreach coef ( 1 2 3 4 5 6 7 8 9 )

        #set logfile = "logs$fitname/log_"`printf "%d_%d" $coef $base`".log"
        set sigma_n_tag = `printf "%d_%d" $coef $base`
        set logfile = "$logdir/log_"$sigma_n_tag".log"
        set val = `printf "%se-%s" $coef $base`

        echo python fit_cogent_data.py --fit $fit --batch --sigma_n $val --tag $fitname"_"$sigma_n_tag $logfile
        python fit_cogent_data.py --fit $fit --batch --sigma_n $val --tag $fitname"_"$sigma_n_tag >& $logfile

    end
end
