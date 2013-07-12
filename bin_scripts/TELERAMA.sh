#!/bin/tcsh

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwlist ath0 scan | grep ESSID
sudo iwconfig ath0 ap any
#sudo iwconfig ath0 ap 00:06:25:51:32:74
#sudo iwconfig ath0 essid "linksys"
#sudo iwconfig ath0 essid "Telerama"
#sudo iwconfig ath0 essid "Crazy Mocha - Shadyside"
sudo dhclient ath0
