#!/bin/tcsh -f 

set key = styFlaN9StuD2
sudo ifconfig eth0 down
sudo ifconfig ath0 up
#sudo iwconfig ath0 essid jlab_guest ap 00:13:5F:55:4F:B0 key s:$key
#sudo iwconfig ath0 essid jlab_guest ap 00:12:44:B8:E3:B0 key s:$key
#sudo iwconfig ath0 essid jlab_guest ap 00:13:7F:C7:E8:F0 key s:$key
#sudo iwconfig ath0 essid jlab_guest ap 00:13:5F:54:B9:00 key s:$key
#sudo iwconfig ath0 essid jlab_guest ap 00:13:5F:57:83:F0 key s:$key
#sudo iwconfig ath0 essid jlab_guest ap 00:13:5F:54:AB:A0 key s:$key
sudo iwconfig ath0 essid jlab_guest ap 00:13:7F:C7:E8:F0 key s:$key
sudo dhclient ath0
