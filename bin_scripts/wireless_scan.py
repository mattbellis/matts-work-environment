#!/usr/bin/env python
import subprocess as sp

import sys

choice = None
if len(sys.argv)>1:
    choice = sys.argv[1]

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

lines = output.split('\n')
for line in lines:
    #print word
    if "Cell" in line:
        words = line.split()
        cell.append(words[1])
        ap.append(words[4])

    if "Channel:" in line:
        channel.append(line.split(":")[1].strip())

    if "ESSID" in line:
        essid.append(line.split(":")[1].strip())

    if "Encryption" in line:
        words = line.split()
        encryption.append(words[1].split(":")[1])

    if "Quality" in line:
        words = line.split()
        strength.append(words[0].split("=")[1].strip())

nsignals = len(cell)
#print cell
#print essid
#print strength
#print ap
#print encryption


print "sudo iwconfig wlan0 essid ### ap ### channel ###"
print "sudo dhclient wlan0"

if choice is not None:
    index = ap.index(choice)
    cmd = ['sudo', 'iwconfig', 'wlan0', 'essid',essid[index],"ap",ap[index],"channel",channel[index]]
    print " ".join(cmd)
    sp.Popen(cmd,0).wait()
else:
    for i in range(0,nsignals):
        output = "%d\t%20s\t%s\t%s\t%s\t%s" % (i,essid[i],strength[i],ap[i],encryption[i],channel[i])
        print output
    print "sudo iwconfig wlan0 essid ### ap ### channel ###"
    print "sudo dhclient wlan0"
