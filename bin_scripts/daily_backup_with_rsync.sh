#!/bin/tcsh 

set today = `date +%b%d"_"%Y`

set logfile = log_daily_$today.log

#set destdir = "/mnt/passport/bellis/backups/" 
set destdir = "/mnt/passport/bellis/backups/" 

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
echo "Backing up daily bellis...." >& logfile 

set output_file = "/mnt/passport/bellis/backups/bellis_daily"

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
      --exclude '*.root' \
      --exclude 'Cache' \
      $output_file >& $logfile


###################################
# Back up my home dir
###################################
echo "Backing up /usr/local...." >& logfile 

set output_file = "/mnt/passport/bellis/backups/usr_local_daily"

echo "Backing up to " $output_file

rsync -CvurltgoD --delete --progress \
      /usr/local \
      $output_file >& $logfile

echo "Finished!"

