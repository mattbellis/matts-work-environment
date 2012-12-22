#!/usr/bin/env tcsh 

set fit = 5

foreach base ( 39 40 41 42 )
    foreach coef ( 1 2 3 4 5 6 7 8 9 )

        #set logfile = "logs$fit/log_"`printf "%d_%d" $coef $base`".log"
        set logfile = "logs$fit/log_"`printf "%d_%d" $coef $base`".log"
        set val = `printf "%se-%s" $coef $base`

        echo python fit_cogent_data.py --fit $fit --batch --sigma_n $val $logfile
        python fit_cogent_data.py --fit $fit --batch --sigma_n $val > $logfile

    end
end
