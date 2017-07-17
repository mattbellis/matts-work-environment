#!/bin/tcsh 

set sides = $1

set printer = 'RB256'
#set printer = 'RB256_color'
#set printer = 'Xerox_Xerox_Phaser_8560DN'
#set printer = 'RB131'

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
        lp -d$printer -o sides=two-sided-long-edge $file
    else if ( $sides == 'single' ) then
        lp -d$printer $file
    endif

end
