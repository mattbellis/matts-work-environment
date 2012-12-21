#/bin/tcsh -f

sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo iwconfig ath0 essid "Hotel Kurfuerstenhof"
sudo iwconfig ath0 ap 00:13:46:73:BC:FC
sudo ifconfig eth0 down
sudo ifconfig ath0 up 
sudo dhclient ath0
sudo ifconfig eth0 down
sudo ifconfig ath0 up 
