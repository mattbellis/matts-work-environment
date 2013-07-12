sudo ifconfig eth0 down
sudo ifconfig ath0 up
#sudo iwconfig ath0 essid "resfac" ap 00:1A:6C:3C:48:30 key s:resfac1234567
sudo iwconfig ath0 essid "resfac" ap 00:1A:6C:3C:43:40 key s:resfac1234567
sudo ifconfig ath0 up
sudo dhclient ath0
