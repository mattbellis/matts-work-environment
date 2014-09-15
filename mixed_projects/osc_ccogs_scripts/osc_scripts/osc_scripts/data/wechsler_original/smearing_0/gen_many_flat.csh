#!/bin/tcsh 

set max = $1

@ i = 0

while ( $i < $max )

    set tag = `printf "%03d" $i`

    echo python generate_flat_ra_dec.py 5000000 "flat_5M_index"$tag".dat"
         python generate_flat_ra_dec.py 5000000 "flat_5M_index"$tag".dat"

    @ i += 1

end
