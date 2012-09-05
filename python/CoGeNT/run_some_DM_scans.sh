#!/usr/bin/env tcsh 


foreach base ( 39 40 41 42 )
    foreach coef ( 1 2 3 4 5 6 7 8 9 )

    set logfile = "logs/log_"`printf "%d_%d" $coef $base`".log"
    set val = `printf "%se-%s" $coef $base`

    echo python fit_cogent_data.py --fit 4 --batch --sigma_n $val $logfile
    python fit_cogent_data.py --fit 4 --batch --sigma_n $val > $logfile
    #echo python fit_cogent_data.py --fit 4 --batch --sigma_n $val > $logfile

    end
end
