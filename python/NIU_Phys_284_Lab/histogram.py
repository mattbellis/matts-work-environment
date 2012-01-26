################################################################################
# Make a histogram and plot it on a figure (TCanvas).
################################################################################

################################################################################
# Import the standard libraries in the accepted fashion.
################################################################################
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

################################################################################
# First thing is read in from a file named mydata.dat
################################################################################

infile = open('mydata.dat')

# Create a variable (array) in which to hold the data.
x = []

################################################################################
# Now read in the data!
################################################################################
for line in infile:

    # !!!!!!! Important!
    # Note that while we are reading in the lines, the following commands are
    # indented _exactly_ the same amount. This is a feature of Python. When
    # You edit this file, make sure you don't adjust the indentation for this
    # section.

    # For our file, we just have a column of numbers. But maybe in the future,
    # you might have more than one column.
    # So we use the ``split" function, which splits a line where ever there
    # are spaces.
    values = line.split()

    # We then add the first value for each line to our ``x" list.
    # Note that we are telling the program to add the value as a ``float", which
    # is just a way that Python stores some numbers.
    x.append(float(values[0]))

################################################################################
# Finished reading in the values
################################################################################

# Let's define some information about the histogram.
num_bins = 10 # Use 100 bins.
lo_range = 0.0 # Set the low and high range.
hi_range = 10.0

# Now define the histogram!
h = plt.hist(x,bins=num_bins,range=(lo_range,hi_range),facecolor='green',histtype='stepfilled')

############################################################################
# Let's format this histogram. 
############################################################################
# Set the y-axis limits, if we so choose.
plt.ylim(0,4)

plt.xlabel('Some text for the x-axis.',fontsize=20)
plt.ylabel('Some other text for the y-axis',fontsize=20)

# Note that we can easily include Latex code
plt.title(r'This is a plot of my data',fontsize=30)

# Need this command to display the figure.
plt.show()

