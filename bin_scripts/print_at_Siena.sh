#!/bin/tcsh 

if ( $# < 4 ) then
    echo
    echo "Usage: print_at_Siena.sh <double or single> <printer (RB256 or RB131, usually)> <# of copies> file(s)"
    echo
    exit
endif

set sides = $1 # single or double
set printer = $2 # Usually RB256 or RB131
set ncopies = $3 # 1,2,3,etc....
#set printer = 'RB256'
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
shift
shift

foreach file ($*)

    echo $file

    if ( $sides == 'double' ) then
        lp -d$printer -n $ncopies -o sides=two-sided-long-edge $file
    else if ( $sides == 'single' ) then
        lp -d$printer -n $ncopies $file
    endif

end
