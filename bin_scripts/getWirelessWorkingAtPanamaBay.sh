sudo ifconfig eth0 down
sudo ifconfig ath0 up
sudo iwlist ath0 scan | grep -v Extra | grep -v Mode | grep -v Freq | grep -v Mb | grep -v WPA | grep -v Enc | grep -v Ciph | grep -v Auth

echo sudo iwconfig ath0 essid PanamaBay
     sudo iwconfig ath0 essid PanamaBay
     #sudo iwconfig ath0 essid linksys
echo sudo iwconfig ath0 ap 00:18:3F:94:3A:C9
     sudo iwconfig ath0 ap 00:18:3F:94:3A:C9
     #sudo iwconfig ath0 ap 00:1A:70:71:99:69

sudo rm /var/run/wpa_supplicant/ath0
sudo /usr/local/sbin/wpa_supplicant -Dmadwifi -iath0 -B -c /home/bellis/wpa_supplicant.Livermore.conf
sudo /usr/local/sbin/wpa_cli identity PanamaBay guest
sudo /usr/local/sbin/wpa_cli password PanamaBay beans
