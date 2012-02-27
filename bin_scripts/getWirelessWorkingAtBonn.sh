#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "gast-bonnet"
sudo iwconfig ath0 ap 00:16:9D:44:8D:71
#sudo iwconfig ath0 ap 00:16:9D:44:8D:11
sudo dhclient ath0
