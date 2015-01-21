#!/bin/tcsh 
sudo ifconfig eth0 down
sudo ifconfig ath0 up

#sudo iwconfig ath0 essid "jlab_secure"
#sudo iwconfig ath0 ap 00:13:5F:54:B9:01
#sudo iwconfig ath0 ap 00:13:5F:55:4F:B1
#sudo iwconfig ath0 ap 00:12:44:B8:E3:B1
#sudo iwconfig ath0 ap 00:13:5F:54:A8:71
#sudo iwconfig ath0 ap 00:12:44:BC:E3:90
#sudo iwconfig ath0 ap 00:13:5F:58:B8:E0
#sudo iwconfig ath0 ap 00:13:5F:59:4F:90
#sudo iwconfig ath0 ap 00:13:5F:58:A8:50
#sudo iwconfig ath0 ap 00:13:5F:54:B0:71
#sudo iwconfig ath0 ap any
##
sudo iwconfig ath0 essid "jlab_guest" ap 00:13:5F:57:7D:C0 key s:wed8nkmpjazEJ
#sudo iwconfig ath0 ap  00:12:44:B8:E3:B0
#sudo iwconfig ath0 ap  00:13:5F:55:4F:B0
#sudo iwconfig ath0 ap  00:13:5F:54:A8:70
#sudo iwconfig ath0 ap  00:13:5F:54:B0:70
#sudo iwconfig ath0 essid resfac

########sudo /usr/local/bin/start-jlab-wpa.pl
########sudo rm /var/run/wpa_supplicant/ath0
########sudo /usr/local/sbin/wpa_supplicant -Dmadwifi -iath0 -c /etc/wpa_supplicant/wpa_supplicant.conf.JLAB -B
########sudo /usr/local/sbin/wpa_cli identity jlab_secure $1
########sudo /usr/local/sbin/wpa_cli password jlab_secure $2


#sudo iwconfig ath0 essid "resfac" key s:resfac1234567
#sudo iwconfig ath0 essid "jlab_secure" ap 00:13:5F:54:A8:71 key s:ibwotRR98atLD
#sudo iwconfig ath0 essid "jlab_secure" ap 00:13:5F:54:A8:71 key s:ibwotRR98atLD
#sudo iwconfig ath0 ap 00:1A:6C:3C:48:50
#sudo iwconfig ath0 essid "jlab_guest"
#sudo iwconfig ath0 ap 00:13:5F:54:A8:70
#sudo iwconfig ath0 key s:Wftu2mWsftuRR
sudo ifconfig ath0 up
sudo dhclient ath0

