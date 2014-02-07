#!/usr/bin/env python

###################################################
# Open a file for writing
###################################################
out_file = open('test_output.txt', 'w+')

# Fill the file with some numbers
for x in range(0,10):
  output = "%d\t" % (x)
  for y in range(1,11):
    output += "%f  " % (float(x)/(2*y))

  # Don't forget the end of line.
  output += "\n"

  # Write the output to the file for each line.
  out_file.write(output)

# Close the file
out_file.close()

###################################################
# Open a file for reading
###################################################
in_file = open('test_output.txt', 'r')
# Print out each line in file.
for line in in_file:
  print line

in_file = open('test_output.txt', 'r')
# Print out 7th entry in each line.
for line in in_file:
  print line.split()[6]

