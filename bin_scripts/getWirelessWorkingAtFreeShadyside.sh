#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "Free Wireless Shadyside" ap 00:18:0A:01:44:B8
sudo ifconfig ath0 up 
sudo dhclient ath0
