#!/bin/tcsh

killall firefox-bin

set file = $HOME/.mozilla/firefox/default.ucu/lock
#ls $file
if ( -l $file ) then
  echo Removing lock file......
  rm $file
endif
