#!/bin/tcsh 

#mcstudy_LambdaC_ntp2_pass0
foreach ntp ( 1 2 )

    rm temp.file >& /dev/null
    echo mcstudy_LambdaC_ntp"$ntp"_pass0 
    ls mcstudy_LambdaC_ntp"$ntp"_pass0 > temp.file
    #cat temp.file

    foreach p_or_e ( "pure" "embed" )
        foreach num(0 10 20 30 40 50 60 70 80 90 100)

        echo " ---------- "
        echo $ntp $p_or_e  $num
        if ( $p_or_e == "pure" ) then
            grep -v "embed" temp.file | grep -v fixed | grep sig"$num""_" | wc -l 
        else
            grep    $p_or_e temp.file | grep -v fixed | grep sig"$num""_" | wc -l 
        endif
            
        end

        foreach num (0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36)

            echo " ---------- "
            echo $ntp $p_or_e  $num fixed
            if ( $p_or_e == "pure" ) then
                grep -v "embed" temp.file | grep fixed | grep sig"$num""_" | wc -l 
            else
                grep    $p_or_e temp.file | grep fixed | grep sig"$num""_" | wc -l 
            endif

        end
    end
end

