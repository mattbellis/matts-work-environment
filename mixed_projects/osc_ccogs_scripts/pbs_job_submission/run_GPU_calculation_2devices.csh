#!/bin/csh

set BIN_DIR = './'
set executable = $BIN_DIR/'angular_correlation'

set file_tag = $1

set data = $2
set flat = $3

################################################################################
# Read in data assuming arc minutes. (-m)
# Even-spaced binning (-l 0)
# Bin width of 1.0 (-w 1.0)
# Low-edge of 1st bin is 1 arg min. (-L 1.00)
################################################################################
#set global_params = '-w 1.0 -L 1.00 -l 0 -m'
#set tag = 'evenbinning_GPU'

################################################################################
# Read in data assuming arc minutes. (-m)
# Log binning (base e) (-l 1)
# Bin width of 0.05 (-w 0.05)
# Low-edge of 1st bin is 1 arg min. (-L 1.00)
################################################################################
#set global_params = '-w 0.05 -L 1.00 -l 1 -m'
#set tag = 'logbinning_GPU'
set global_params = '-w 0.05 -L 1.00 -l 1 -s -S'
#set global_params = '-w 0.05 -L 1.00 -l 1 -s'
set tag = "logbinning_GPU"

################################################################################
# Read in data assuming arc minutes. (-m)
# Log10 binning (base 10) (-l 2)
# Bin width of 0.02 (-w 0.02)
# Low-edge of 1st bin is 1 arg min. (-L 1.00)
################################################################################
#set global_params = '-w 0.02 -L 1.00 -l 2 -m'
#set tag = 'log10binning_GPU'


echo "#####################"
echo "Submitting DR calculation on device 0..."
date;
time $executable $data $flat $global_params -o "$tag"_"$file_tag"_data_flat_arcsec.dat -d 0 #>& log"$tag"_"$file_tag"_dr.log & 
echo "#####################"
echo "Submitting DD calculation on device 1..."
date;
time $executable $data $data $global_params -o "$tag"_"$file_tag"_data_data_arcsec.dat -d 1 #>& log"$tag"_"$file_tag"_dd.log 
echo "#####################"
echo "Submitting RR calculation on device 1..."
date;
time $executable $flat $flat $global_params -o "$tag"_"$file_tag"_flat_flat_arcsec.dat -d 1 #>& log"$tag"_"$file_tag"_rr.log 
echo "Finished with device 1 stuff"
date;
wait
echo "Finished with everything!"
date;
echo "Completed!"
date;

