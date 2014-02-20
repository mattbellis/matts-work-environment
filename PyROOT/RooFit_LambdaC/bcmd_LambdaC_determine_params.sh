#!/bin/tcsh

################################################################################
# Set the new ROOTSYS and everything else that follows.
################################################################################
setenv ROOTSYS  ~bellis/LeptBc_AWG120_bellis/root
setenv LD_LIBRARY_PATH ${ROOTSYS}/lib/:${LD_LIBRARY_PATH}
setenv PYTHONPATH ${ROOTSYS}/lib

################################################################################
# Use the KIPAC python.
################################################################################
setenv PYTHON_LOCATION /afs/slac/g/ki/software/python/2.5.4/

setenv LD_LIBRARY_PATH ${PYTHON_LOCATION}/lib/:${LD_LIBRARY_PATH}

alias python $PYTHON_LOCATION/bin/python
setenv MY_PYTHON ${PYTHON_LOCATION}/bin/python

################################################################################
# Write these out for diagnostics.
################################################################################
echo $ROOTSYS
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

################################################################################
# Execute the script in the directory where everything lives.
################################################################################
cd /a/sulky61/AWG120/LeptBc/bellis/PyRoot/RooFit_LambdaC/

################################################################################
# Run the command with whatever command line options we need.
################################################################################
python run_a_set_of_fits.py --my-python $MY_PYTHON --ntp 1 --baryon LambdaC --pass 0 --step 0 
#python run_a_set_of_fits.py --my-python $MY_PYTHON --ntp 1 --baryon LambdaC --pass 0 --step 1 
#python run_a_set_of_fits.py --my-python $MY_PYTHON --ntp 2 --baryon LambdaC --pass 0 --step 0 
#python run_a_set_of_fits.py --my-python $MY_PYTHON --ntp 2 --baryon LambdaC --pass 0 --step 1 

