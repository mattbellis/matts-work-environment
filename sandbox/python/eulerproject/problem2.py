#!/usr/bin/env python

# Initialize the Fib sequence
numbers = [0, 1, 1]

# Initialize the sum
sum = 0

# Loop over this while the number<4e6 condtion is met
while numbers[2] < 4000000:

  # Store some of the numbers so we can keep track
  # of the two previous numbers when we calculate the
  # next term in the sequence
  temp = [numbers[1], numbers[2]]

  # Calculate the next number
  numbers[2] += numbers[1]

  # Swap out the numbers 
  numbers[0] = temp[0]
  numbers[1] = temp[1]

  if numbers[2]%2 == 0:
    sum += numbers[2]
    print "fib num: %10d\t\tsum:%10d" % (numbers[2], sum)
