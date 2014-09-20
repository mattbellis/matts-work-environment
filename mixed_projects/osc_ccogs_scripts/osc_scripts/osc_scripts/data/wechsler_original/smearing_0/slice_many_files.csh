#!/bin/tcsh

foreach file(wechsler_gals_index00[1-9].dat wechsler_gals_index0[1-4]*.dat)
    echo $file
    python Z_Slices_Code.py $file
end
