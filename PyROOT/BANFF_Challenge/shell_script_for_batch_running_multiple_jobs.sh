#!/bin/csh

ls -altr

#set inputfile = $1
set nsig = $1
set nbootstrap_samples = $2
set distance = $3
@ lo = $4
@ hi = $5

set workdir = "/scratch/bellis_nn_fits"

set mydir = "~bellis/rootScripts/nearestNeighborFitting/"

date
mkdir -p $workdir
ls -l $workdir
which nn_fit_for_BANFF_Challenge_MaximumLikelihood 

################################################################################
echo "cd'ing to workdir......"
cd $workdir

ls -altr

cp $mydir/bc2p2bg1mc.dat .
cp $mydir/bc2p2bg2mc.dat .
cp $mydir/bc2p2sigmc.dat .
################################################################################


################################################################################
################################################################################
echo "About to start running..........."

@ i = $lo
while ( $i <= $hi )

    set inputfile = "toy_datasets/toy_"$nsig"_"$i".dat"

    cp $mydir/$inputfile .

    set inputfile = `basename $inputfile`

    nn_fit_for_BANFF_Challenge_MaximumLikelihood $inputfile bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat $nbootstrap_samples $distance 

    @ i ++

end

echo "Finished running..........."
################################################################################
################################################################################

# Clean up the directory for the next person
rm -r $workdir/$inputfile
rm -r $workdir/bc2p2bg1mc.dat 
rm -r $workdir/bc2p2bg2mc.dat 
rm -r $workdir/bc2p2sigmc.dat 

