#!/bin/tcsh 

echo "Pushing from luminous to serenity"
rsync -P -r -u -a -v --exclude '*lost+found*' --exclude '/data' /data/* 192.168.6.107:/data
#echo "Pulling from serenity to luminous"
#rsync -P -r -u -a -v --exclude '*lost+found*' --exclude '/data' 192.168.6.107:/data/ /data

