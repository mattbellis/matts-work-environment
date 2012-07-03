#!/bin/tcsh 
sudo ifconfig wlan0 up

##
sudo iwconfig wlan0 essid "wireless" ap 00:02:B3:A5:AF:72 key 1962ab1968
sudo ifconfig wlan0 up
sudo dhclient wlan0

