#!/usr/bin/env python

import sys
import re

import subprocess as sp

import time


################################################################################
# Open the output file in append mode
################################################################################
outfile = open("core_temps.txt",'a')

################################################################################
# Run sensors every 10 seconds
################################################################################
while 1!=0:

  # First grab the current time
  now = time.asctime(time.localtime())
  print now
  outfile.write(now)
  outfile.write("\n")

  # Run sensors, grab the output, and write it to the outfile
  cmd  = ['sensors']
  output = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]

  outfile.write(output)

  # Sleep for 10 seconds before looping around 
  time.sleep(10)

