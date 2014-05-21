#!/bin/tcsh

set my_array = ( "ntp1"  \
"ntp2" )

@ i = 1

while ( $i < 3 )

    echo $i " " $my_array[$i]

    @ i++

end

