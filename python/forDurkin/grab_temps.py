#!/usr/bin/env python

import sys
import re

import subprocess as sp


################################################################################
# Open the output file in append mode
################################################################################
outfile = open("core_temps.txt",'a')

################################################################################
# Run sensors every 10 seconds
################################################################################
while 1!=0:

  cmd  = ['sensors']
  output = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]

  outfile.write(open)


# Holder for the temps: for core 0 and 1 (the array indices)
core_temps = [ [], [] ]

for line in infile:
  #print line
  
  # Make sure it's one of the Core's for which we're looking
  index = -1
  if line.find('Core 0')>=0:
    index = 0
  if line.find('Core 1')>=0:
    index = 1

  #print index

  if index==-1:
    x = 1 # Do nothing
  else: # Parse the line
    this_core_temps = []
    vals = line.split()
    for v in vals:
      #print v
      if v[0]=='+':
        temp = v.split('+')[1]
        decimal_location = temp.find('.')
        temp = temp[0:decimal_location+2]
        #print temp
        this_core_temps.append(float(temp))

    # Store the temps in our main array
    core_temps[index].append(this_core_temps)
        

# Dump out the results
for i,ct in enumerate(core_temps):
  print "Core %d" % (i)
  for c in ct:
    print "\tinstantaneous temp: %4.1f C" % (c[0])
    print "\thigh temp: %4.1f C" % (c[1])
    print "\tcritical temp: %4.1f C" % (c[2])
      

