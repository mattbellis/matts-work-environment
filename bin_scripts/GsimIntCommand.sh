#/bin/tcsh -f 

rm paw.metafile
rm last.kumac
rm gsim.evt.A00

#setenv CLAS_PARMS /home/bellis/latestPARMS
setenv CLAS_PARMS /home/bellis/g11PARMS
#setenv CLAS_PARMS /home/bellis/g1cPARMS

echo $CLAS_PARMS

$CLAS_BIN/gsim_int -ffread $1 -mcin $2 -kine 1 -bosout gsim.evt
