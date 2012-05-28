#!/bin/tcsh 

set today = `date +%b%d"_"%Y`

set logfile = log$today.log

set destdir = "/mnt/externaldrive/bellis/backups/" 

@ max_backups_to_keep = 4

if ( ! -e $destdir ) then
  echo 
  echo "Exiting!"
  echo "Destination directory does not exist"
  echo
  exit(1)
endif

rm $logfile >& /dev/null

###################################
# Back up my home dir
###################################
@ num_backups =  `ls -ld $destdir/bellis* | grep weekly | wc -l | awk '{print $1}'`

if ( $num_backups > $max_backups_to_keep ) then
 @ diff = $num_backups - $max_backups_to_keep 
 echo "Grabbing old backups to remove..."
 echo `ls -dtr $destdir/bellis* | grep weekly | head -n $diff`
 rm -rf `ls -dtr $destdir/bellis* | grep weekly | head -n $diff`
endif

echo "Backing up bellis...." >& logfile 

set output_file = "/mnt/externaldrive/bellis/backups/bellis_weekly_$today "

echo "Backing up to " $output_file


rsync -CvurltgoD --delete --progress \
      /home/bellis \
      --exclude 'Desktop/' \
      --exclude 'bellis/src/' \
      --exclude '.mp3' \
      --exclude 'mp3*/' \
      --exclude '*.avi' \
      --exclude '*.mpg' \
      --exclude '*.mpeg' \
      --exclude '*.mov' \
      --exclude 'Cache/' \
      $output_file >& $logfile


echo "Backing up /usr/local...." >& logfile 



###################################
# Back up my home dir
###################################
@ num_backups =  `ls -ld $destdir/usr_local* | grep weekly | wc -l | awk '{print $1}'`

if ( $num_backups > $max_backups_to_keep ) then
 @ diff = $num_backups - $max_backups_to_keep 
 echo "Grabbing old backups to remove..."
 echo `ls -dtr $destdir/usr_local* | grep weekly | head -n $diff`
 rm -rf `ls -dtr $destdir/usr_local* | grep weekly | head -n $diff`
endif

echo "Backing up /usr/local...." >& logfile 

set output_file = "/mnt/externaldrive/bellis/backups/usr_local_weekly_$today "

echo "Backing up to " $output_file

rsync -CvurltgoD --delete --progress \
      /usr/local \
      $output_file >& $logfile

echo "Finished!"

