#/bin/tcsh -f


# Try these sometime
#iwconfig ath0 mode managed rate auto rts off frag off key off enc off
#iwconfig ath0 txpower auto  # seems to power up transmitter
#iwconfig ath0 essid any     # seems to power up transmitter
#ifconfig ath0 up            # initiates active scan for access point

# Or this
#add default gw 192.168.123.254

# Or this
#Much more common is to get a valid DHCP address but no nameserver, the service that maps domain names such as example.com to Internet addresses. If this is the case, your network will be up and you'll be able to get to IP addresses or hosts specified in /etc/hosts, but you'll hang any time you try to reference any other host by name.

#On Linux, user cat /etc/resolv.conf to find out what DNS information you've been given. It may be wrong in some fairly obvious way. For instance, one hotel listed 0.0.0.2 as the nameserver, but DHCP had assigned our machines addresses of the form 10.0.0.18. After editing /etc/resolv.conf to change the nameserver's address to 10.0.0.2, everything worked fine.

#If there's no obvious error in resolv.conf, try using the DNS server you use at home. Pull out the resolv.conf you copied at home and install it in /etc. It may help.


sudo ifconfig eth0 down
sudo ifconfig ath0 up 
#sudo iwconfig ath0 essid "Coco's Cupcake Cafe" ap 00:12:0E:5A:51:5C key off
sudo iwconfig ath0 essid "Crazy Mocha - Shadyside" ap 00:12:0E:54:B5:BF key off
#sudo iwconfig ath0 essid "Free Wireless Shadyside" ap 00:18:0A:01:44:B8 key off
#sudo iwconfig ath0 essid "Free Wireless Shadyside" ap 00:18:0A:01:44:B4 key off
sudo ifconfig ath0 up 
sudo dhclient ath0
