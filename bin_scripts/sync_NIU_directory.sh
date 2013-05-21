#!/bin/tcsh 

if ( $# == 0 ) then
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif


if ( $1 == "push" ) then
    rsync -P -r -u -a -v \
        --exclude '*.npy' \
        --exclude '*.ig' \
        --exclude 'Event_*' \
        --exclude '*/core' \
        --exclude '*talk*.pdf' \
        --exclude '*lecture*.pdf' \
        --exclude '*chapter*.pdf' \
        --exclude '*/data_skims/*' \
        --exclude '*/processing-2.0b3/*' \
        /home/bellis/printer_drivers \
        /home/bellis/bluehost_staging \
        /home/bellis/Stanford \
        /home/bellis/Siena \
        /home/bellis/wallpapers \
        /home/bellis/eBooks \
        /home/bellis/Work \
        /home/bellis/papers \
        /home/bellis/latex_stuff \
        /home/bellis/BaBar \
        /home/bellis/Jobs  \
        /home/bellis/stuff \
        /home/bellis/NIU \
        /home/bellis/Talks \
        /home/bellis/sketchbook \
        74.76.135.220:/home/bellis
        #mattbellis.dyndns-home.com:/home/bellis

else if ( $1 == "pull" ) then
    rsync -P -r -u -a -v --exclude '*.npy'  --exclude '*.ig' --exclude 'Event_*'   --exclude '*/core' --exclude '*talk*.pdf' --exclude '*lecture*.pdf' --exclude '*chapter*.pdf' --exclude '*/data_skims/*' --exclude '*/processing-2.0b3/*' mattbellis.dyndns-home.com:/home/bellis/{Talks,sketchbook,Work,NIU,BaBar,Jobs,stuff,papers,latex_stuff,bluehost_staging,Stanford,Siena,eBooks,wallpapers,printer_drivers} \
        /home/bellis

else
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif
