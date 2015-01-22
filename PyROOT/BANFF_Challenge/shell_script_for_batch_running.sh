#!/bin/csh

#date
#mkdir -p /scratch/bellis_dilTuple_SP1005
#ls -l /scratch/bellis_dilTuple_SP1005
#which DileptonCPTApp
#../bin/$BFARCH/DileptonCPTApp tcl_run_dilep/SP1005/run_dilTuple_SJM_SP1005_Run3_1.tcl
#echo Ran to completion
## What's in the directory
#ls -ltr /scratch/bellis_dilTuple_SP1005
## Copy over the files
#cp -p /scratch/bellis_dilTuple_SP1005/dilTuple_SJM_SP1005_Run3_1.root rootfiles/SP1005/.
#ls -l rootfiles/SP1005/dilTuple_SJM_SP1005_Run3_1.root
## Clean up the files
#rm -r /scratch/bellis_dilTuple_SP1005/dilTuple_SJM_SP1005_Run3_1.root

ls -altr

set inputfile = $1
set nbootstrap_samples = $2
set distance = $3

set workdir = "/scratch/bellis_nn_fits"

set mydir = "~bellis/rootScripts/nearestNeighborFitting/"

date
mkdir -p $workdir
ls -l $workdir
which nn_fit_for_BANFF_Challenge_MaximumLikelihood 

echo "cd'ing to workdir......"
cd $workdir

ls -altr

cp $mydir/bc2p2bg1mc.dat .
cp $mydir/bc2p2bg2mc.dat .
cp $mydir/bc2p2sigmc.dat .
cp $mydir/$inputfile .

echo "About to start running..........."

set inputfile = `basename $inputfile`
set logfile = "log_nbs$nbootstrap_samples"_r"$distance"_$inputfile

nn_fit_for_BANFF_Challenge_MaximumLikelihood $inputfile bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat $nbootstrap_samples $distance 

echo "Finished running..........."

# Clean up the directory for the next person
rm -r $workdir/$inputfile
rm -r $workdir/bc2p2bg1mc.dat 
rm -r $workdir/bc2p2bg2mc.dat 
rm -r $workdir/bc2p2sigmc.dat 

