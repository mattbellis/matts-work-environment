#!/usr/bin/env python

import subprocess as sp

cmd = ['sudo', 'ifconfig', 'eth0', 'down']
sp.Popen(cmd,0).wait()

cmd = ['sudo', 'ifconfig', 'wlan0', 'up']
sp.Popen(cmd,0).wait()

cmd = ['sudo', 'iwlist', 'wlan0', 'scan']
output = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]

cell = []
essid = []
channel = []
ap = []
encryption = []
strength = []

words = output.split()
for i,word in enumerate(words):
    #print word
    if "Cell" == word:
        cell.append(words[i+1])
        ap.append(words[i+4])

    if "Channel:" in word:
        #print word
        channel.append(word.split(":")[1].strip())

    if "ESSID" in word:
        essid.append(word.split(":")[1].strip())

    if "Encryption" == word:
        encryption.append(words[i+1].strip())

    if "Quality" in word:
        strength.append(word.split("=")[1].strip())

nsignals = len(cell)
#print cell
#print essid
#print strength
#print ap
#print encryption

for i in range(0,nsignals):
    output = "%d\t%20s\t%s\t%s\t%s\t%s" % (i,essid[i],strength[i],ap[i],encryption[i],channel[i])
    print output

print "sudo iwconfig wlan0 essid ### ap ### channel ###"
print "sudo dhclient wlan0"

