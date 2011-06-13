#!/usr/bin/env python

# Number to factor
x = 600851475143

# Loop over this while the number<4e6 condtion is met
not_found = True
divisor = 2
#while not_found:
while x>10:

  #print "Trying %d" % (divisor)

  if x%divisor == 0:
    x /= divisor
    print "\t\t\t%d %d" % (divisor, x)
    #not_found = False

  divisor += 1
