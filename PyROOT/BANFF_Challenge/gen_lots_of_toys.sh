#!/bin/tcsh 

set nsig = 75

@ i = 0

while ( $i < 1000 )

    echo $i " ----------- "
    ./mix_up_a_test_sample.py $nsig toy_datasets/toy_$nsig"_"$i.dat

    @ i++

end
