#!/bin/tcsh 

set backupfile = BellisBackup`date +%m%d%y`.tar
set logfile = BellisBackup`date +%m%d%y`.log

echo $backupfile
echo $logfile

rm $backupfile $logfile >& /dev/null

tar -cvf $backupfile /home/bellis/personal/stuff >& $logfile
tar -cvf $backupfile /home/bellis/.vimrc >& $logfile
tar -rvf $backupfile /home/bellis/.cshrc >>& $logfile
tar -rvf $backupfile /usr/src/linux/.config >>& $logfile
tar -rvf $backupfile /home/bellis/.rootrc >>& $logfile
#tar -rvf $backupfile /home/bellis/rootStuff/ >>& $logfile
tar -rvf $backupfile /home/bellis/latexStuff/ >>& $logfile
tar -rvf $backupfile /home/bellis/CodeTestingGround/ >>& $logfile
tar -rvf $backupfile /home/bellis/Paperwork/ >>& $logfile
tar -rvf $backupfile /home/bellis/XfigStuff/*/*.fig >>& $logfile
#tar -rvf $backupfile /home/bellis/.mozilla >>& $logfile
#tar -rvf $backupfile /home/bellis/.mozilla-thunderbird/ >>& $logfile
tar -rvf $backupfile /home/bellis/bin/*.sh >>& $logfile
tar -rvf $backupfile `find /home/bellis/rootStuff/* | grep '\.h'` >>& $logfile
tar -rvf $backupfile `find /home/bellis/rootStuff/* | grep '\.C'` >>& $logfile
tar -rvf $backupfile `find /home/bellis/rootStuff/* | grep '\.cc'` >>& $logfile
tar -rvf $backupfile `find /home/bellis/rootStuff/*/* | grep '\.py'` >>& $logfile
tar -rvf $backupfile `find /home/bellis//* | grep '\.h' | grep -v hddm` >>& $logfile
tar -rvf $backupfile `find /home/bellis//* | grep '\.cc'` >>& $logfile
tar -rvf $backupfile `find /home/bellis//* | grep '\.C'` >>& $logfile
tar -rvf $backupfile `find /home/bellis//* | grep 'Makefile'` >>& $logfile
tar -rvf $backupfile `find /home/bellis/* | grep '\.tex'` >>& $logfile
tar -rvf $backupfile `find /home/bellis/* | grep '\.py'` >>& $logfile

gzip $backupfile >>& $logfile
