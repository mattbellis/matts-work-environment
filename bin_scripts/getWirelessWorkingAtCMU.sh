#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "CMU" ap 00:14:6A:5B:92:20 key off
#sudo iwconfig ath0 ap any
#sudo iwconfig ath0 essid "CMU" ap 00:14:6A:5B:66:70 key off
#sudo iwconfig ath0 essid "CMU" ap 00:14:6A:5B:9B:90 key off
#sudo iwconfig ath0 essid "CMU" key open
sudo ifconfig ath0 up 
sudo dhclient ath0
