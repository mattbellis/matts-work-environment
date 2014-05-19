#!/usr/bin/env python

# Loop from 0 to 10 in steps of 1
print "Using range(0,10)"
for x in range(0,10):
  print x
print "\n"

# Loop from 1 to 10 in steps of 2
print "Using range(1,10,2)"
for x in range(1,10,2):
  print x
print "\n"

# Loop from 0 to 10 in steps of 1
print "Using xrange(10)"
for x in xrange(10):
  print x
print "\n"

# Loop over a list
print "Iterating over a list"
words = ['Toad', 'the', 'wet', 'sprocket']
for w in words:
  print w
print "\n"

# Enumerate over a list
print "Enumerating over a list"
words = ['Toad', 'the', 'wet', 'sprocket']
for n,w in enumerate(words):
  print "Word #%d: %s" % (n, w)
print "\n"


