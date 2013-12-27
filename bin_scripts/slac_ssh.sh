#!/bin/tcsh 

set machine = "yakut16"
if ( $1 != "" ) then
  set machine = "yakut"$1
endif

echo $machine
ssh -X -v bellis@$machine.slac.stanford.edu
