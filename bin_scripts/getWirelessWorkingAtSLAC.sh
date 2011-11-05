#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
#sudo iwconfig ath0 essid "visitor" ap 00:11:92:DB:2B:50
sudo iwconfig ath0 essid "visitor" ap 00:11:93:3D:70:10
sudo dhclient ath0
