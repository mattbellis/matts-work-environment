#!/bin/sh
sudo ifconfig eth1 down
sudo ifconfig wlan0 up
#wlanconfig ath0 create wlandev wifi0 wlanmode sta
#wpa_supplicant -Bw -Dmadwifi -iath0 -c$1


#sudo wpa_supplicant -w -Dmadwifi -iwlan0 -c$1
sudo wpa_supplicant -iwlan0 -c$1

