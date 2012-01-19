#!/bin/tcsh 

set nbkg1 = 1000
set nbkg2 = 100
set nsig = 75

@ i = 0

while ( $i < 1000 )

    echo $i " ----------- "
    set infile = "toy_datasets/toy_$nbkg1"_"$nbkg2"_"$nsig"_"$i.dat"
    nn_fit_for_BANFF_Challenge_MaximumLikelihood $infile bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.0005

    @ i++

end
