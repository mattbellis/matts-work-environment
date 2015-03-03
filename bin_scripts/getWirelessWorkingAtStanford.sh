#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "Stanford" ap 00:11:93:1F:51:00
sudo dhclient ath0
