#!/bin/tcsh 

#set dir_to_sync = "Talks"
#set dir_to_sync = "NIU"

if ( $# == 0 ) then
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif


if ( $1 == "push" ) then
    rsync -P -r -u -a -v /home/bellis/BaBar /home/bellis/NIU /home/bellis/Talks mattbellis.dyndns-home.com:/home/bellis
    #rsync -P -r -u -a -v /home/bellis/NIU /home/bellis/Talks mattbellis.dyndns-home.com:/home/bellis
else if ( $1 == "pull" ) then
    rsync -P -r -u -a -v mattbellis.dyndns-home.com:/home/bellis/{Talks,NIU,BaBar} /home/bellis
    #rsync -P -r -u -a -v mattbellis.dyndns-home.com:/home/bellis/{Talks,NIU} /home/bellis
else
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif
