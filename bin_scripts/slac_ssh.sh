#!/bin/tcsh 

set machine = "yakut16"
if ( $1 != "" ) then
  set machine = "yakut"$1
endif

ssh bellis@$machine.slac.stanford.edu
