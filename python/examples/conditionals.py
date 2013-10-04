#!/usr/bin/env python

import sys

# Read in the first two command line parameters
x = int(sys.argv[1])
y = int(sys.argv[2])

# Test which is greater.
if x > y:
  print "x is greater than y"
elif y > x:
  print "y is greater than x"
else: 
  print "x equals y"

# Explicitly use True/False
if (x>y) == True:
  print "x>y is True"
elif (x>y) == False:
  print "x>y is False"
