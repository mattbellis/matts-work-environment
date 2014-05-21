#!/bin/tcsh 

#foreach nboots( 0 10 100 1000 )
foreach nboots( 0 10 100 1000 )
    foreach distance ( 0.005 0.010 0.020 )
    #foreach distance ( 0.020 )

        echo " --------------- "
        echo " --------------- "
        echo $nboots " " $distance
        echo " --------------- "

        set distance_tag = `echo $distance | awk '{print $1*1000}'`

        ./plot_significances_for_a_study.py \
            logfiles/collated_logfile_nsig0_nbs"$nboots"_r"$distance".log \
            logfiles/collated_logfile_nsig75_nbs"$nboots"_r"$distance".log \
            "toy"_nbs"$nboots"_r"$distance_tag" \
            batch

        if ( ( $nboots == 100 || $nboots == 1000 ) && \
             ( $distance == 0.005 || $distance == 0.010 ) ) then

            ./plot_significances_for_a_study.py \
                logfiles/collated_logfile_nsig0_nbs"$nboots"_r"$distance".log \
                logfiles/collated_logfile_nbs"$nboots"_r"$distance".log \
                "data"_nbs"$nboots"_r"$distance_tag" \
                batch

        endif

    end
end
