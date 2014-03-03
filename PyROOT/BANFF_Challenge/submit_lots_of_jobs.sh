#!/bin/tcsh 

foreach nsig ( 0 75 )
    foreach nboots( 2 10 50 100 500 1000 )
        foreach distance ( 0.005 0.007 0.010 0.015 0.05 )

            @ lo = 0
            @ hi = 1000
            @ step = 200

            if ( $nboots == 50 ) then
                @ step = 100
            else if ( $nboots == 100 ) then
                @ step = 100
            else if ( $nboots == 500 ) then
                @ step = 20
            else if ( $nboots == 1000 ) then
                @ step = 10
            endif

            @ i = $lo

            while ( $i < $hi ) 

                @ j = $i + $step - 1

                echo ./bsub_cmds.sh $nsig $nboots $distance $i $j
                     ./bsub_cmds.sh $nsig $nboots $distance $i $j

                 @ i += $step

            end

        end
    end
end
