#!/usr/bin/env python

# Can add strings
x = 'Monty'
y = 'Python'
name = x + ' ' + y
print name

# Can do equivalent of sprintf
name = "%s %s" % (x, y)
print name

# Searching in a string
if 'Mo' in name:
  print "Found Mo!"

# Searching in a string
print name.find('on')
print name.find('no')
