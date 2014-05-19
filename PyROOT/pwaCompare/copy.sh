#!/bin/tcsh -f

#set fit = $1
#set type = $2
set type = "coupled"
#set fit = "covariant_0123_Fit125"
#set fit = "covariant_123_1357DeltaRho"
#set fit = "covariant_0123_135DeltaRho"
set fit = "covariant_123_135DeltaRho"

  set destdir = "rootFiles"
  
  if ( ! -e $destdir ) then
    mkdir $destdir
  endif

  echo pwa@erwin.phys.cmu.edu:/raid13/pwa/g1c_2pi/fits/event_fits/$type/$fit/rootFiles/combinedFiles/
  scp "pwa@erwin.phys.cmu.edu:/raid13/pwa/g1c_2pi/fits/event_fits/$type/$fit/rootFiles/combinedFiles/*_14*-*.root" $destdir/.



