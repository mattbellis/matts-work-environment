#!/bin/tcsh 

set sides = $1

if ( $sides == 'double' || $sides == 'single' ) then
    echo "Printing..."
else
    echo "First argument must be single or double!"
    exit(-1)
endif

shift

foreach file ($*)

    echo $file

    if ( $sides == 'double' ) then
        lp -dnicadd_color -o sides=two-sided-long-edge $file
    else if ( $sides == 'single' ) then
        lp -dnicadd_color $file
    endif

end
