#!/bin/tcsh 

set machine = "yakut10"
if ( $1 != "" ) then
  set machine = $1
endif

ssh bellis@$machine.slac.stanford.edu
