#!/bin/tcsh 
sudo ifconfig wlan0 up

##
sudo iwconfig wlan0 essid "DNP08secure" ap 00:1D:E5:8C:4F:B0 key s:Type-III-nova
sudo ifconfig wlan0 up
sudo dhclient wlan0

