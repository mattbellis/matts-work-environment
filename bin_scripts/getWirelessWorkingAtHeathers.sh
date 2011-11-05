#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "pandanet" ap 00:12:17:CF:19:7D
sudo dhclient ath0
