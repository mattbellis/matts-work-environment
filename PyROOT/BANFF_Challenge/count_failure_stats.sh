#!/bin/tcsh 

#foreach nboots( 0 10 100 1000 )
foreach nboots( 1000 )
    foreach distance ( 0.005 0.010 0.020 )
    #foreach distance ( 0.020 )

        echo " --------------- "
        echo " --------------- "
        echo $nboots " " $distance
        echo " --------------- "

        ./count_failure_stats.py logfiles/collated_logfile_nsig0_nbs"$nboots"_r"$distance".log logfiles/collated_logfile_nsig75_nbs"$nboots"_r"$distance".log

    end
end
