#!/bin/tcsh 

set nbkg1 = 1000
set nbkg2 = 100
set nsig = 75

@ i = 0

while ( $i < 1000 )

    echo $i " ----------- "
    ./mix_up_a_test_sample.py $nbkg1 $nbkg2 $nsig toy_datasets/toy_$nbkg1"_"$nbkg2"_"$nsig"_"$i.dat

    @ i++

end
