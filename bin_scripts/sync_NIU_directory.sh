#!/bin/tcsh 

if ( $# == 0 ) then
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif


if ( $1 == "push" ) then
    rsync -P -r -u -a -v \
        --exclude '*.npy' \
        --exclude '*.ig' \
        --exclude '*JLab*' \
        --exclude 'Event_*' \
        --exclude '*/core' \
        --exclude '*talk*.pdf' \
        --exclude '*lecture*.pdf' \
        --exclude '*chapter*.pdf' \
        --exclude '*/data_skims/*' \
        --exclude '*/processing-2.0b3/*' \
        --exclude '*/BaBar/*' \
        /home/bellis/printer_drivers \
        /home/bellis/bluehost_staging \
        /home/bellis/Stanford \
        /home/bellis/Siena \
        /home/bellis/wallpapers \
        /home/bellis/eBooks \
        /home/bellis/Work \
        /home/bellis/papers \
        /home/bellis/latex_stuff \
        #/home/bellis/BaBar \
        /home/bellis/Jobs  \
        /home/bellis/stuff \
        /home/bellis/NIU \
        /home/bellis/Talks \
        /home/bellis/sketchbook \
        bellis@mattbellis.dyndns-home.com:/home/bellis
            #/home/bellis/BaBar \
        #bellis@192.168.7.23:/home/bellis
            #74.76.135.220:/home/bellis

else if ( $1 == "pull" ) then
    #rsync -P -r -u -a -v --exclude '*.npy'  --exclude '*JLab*'  --exclude '*.ig' --exclude 'Event_*'   --exclude '*/core' --exclude '*talk*.pdf' --exclude '*lecture*.pdf' --exclude '*chapter*.pdf' --exclude '*/data_skims/*' --exclude '*/processing-2.0b3/*' bellis@192.168.7.23:/home/bellis/{Talks,sketchbook,Work,NIU,BaBar,Jobs,stuff,papers,latex_stuff,bluehost_staging,Stanford,Siena,eBooks,wallpapers,printer_drivers} \
    rsync -P -r -u -a -v --exclude '*.npy'  --exclude '*JLab*'  --exclude '*.ig' --exclude 'Event_*'   --exclude '*/core' --exclude '*talk*.pdf' --exclude '*lecture*.pdf' --exclude '*chapter*.pdf' --exclude '*/data_skims/*' --exclude '*/processing-2.0b3/*' bellis@mattbellis.dyndns-home.com:/home/bellis/{Talks,sketchbook,Work,NIU,BaBar,Jobs,stuff,papers,latex_stuff,bluehost_staging,Stanford,Siena,eBooks,wallpapers,printer_drivers} \
#rsync -P -r -u -a -v --exclude '*.npy'  --exclude '*BaBar*'  --exclude '*JLab*'  --exclude '*.ig' --exclude 'Event_*'   --exclude '*/core' --exclude '*talk*.pdf' --exclude '*lecture*.pdf' --exclude '*chapter*.pdf' --exclude '*/data_skims/*' --exclude '*/processing-2.0b3/*' bellis@192.168.7.23:/home/bellis/{Talks,sketchbook,Work,NIU,BaBar,Jobs,stuff,papers,latex_stuff,bluehost_staging,Stanford,Siena,eBooks,wallpapers,printer_drivers} \
        /home/bellis

else
    echo "First argument must be 'pull' or 'push'"
    exit(-1)
endif
