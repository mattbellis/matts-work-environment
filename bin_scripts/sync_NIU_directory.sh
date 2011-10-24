#!/bin/tcsh 

if ( $# == 0 ) then
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif


if ( $1 == "push" ) then
    rsync -P -r -u -a -v /home/bellis/bluehost_staging /home/bellis/papers /home/bellis/BaBar /home/bellis/Jobs  /home/bellis/stuff /home/bellis/NIU /home/bellis/Talks mattbellis.dyndns-home.com:/home/bellis
else if ( $1 == "pull" ) then
    rsync -P -r -u -a -v mattbellis.dyndns-home.com:/home/bellis/{Talks,NIU,BaBar,Jobs,stuff,papers,bluehost_staging} /home/bellis
else
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif
