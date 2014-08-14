#!/bin/tcsh 

foreach nsig ( 0 75 )
    foreach nboots( 2 10 50 100 500 1000 )
        foreach distance ( 0.005 0.007 0.010 0.015 0.05 )

            echo $nsig" "$nboots" "$distance
            set summary_logfile = "logfiles/summary_logfile__nsig"$nsig"_nbs"$nboots"_r"$distance".log"

            rm -rf $summary_logfile >& /dev/null


            grep -C12 'nfit: 0	# k3:' logfiles/log_nsig"$nsig"_nbs"$nboots"_r"$distance"* > $summary_logfile

        end
    end
end
