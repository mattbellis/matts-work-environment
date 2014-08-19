#!/bin/tcsh 

set zlo = '0.200'
set zhi = '0.225'

@ i = 0

while ( $i < 50 )

    set index = `printf "%03d" $i`

    set data = z-slice_index"$index"_"$zlo"_to_"$zhi".dat
    set random = flat_5M_index"$index".dat

    set tag = "z_smearing_index"$index"_z_"$zlo"to"$zhi

    echo python build_submission_files_z_smeared.py $tag $data $random
         python build_submission_files_z_smeared.py $tag $data $random

    @ i += 1
end
