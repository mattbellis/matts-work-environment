import matplotlib.pylab as plt
import numpy as np
import sys

# Define your file name.
infilename = sys.argv[1]
infile = open(infilename,'r')

# Holders for your x and y points. Note that these are just
# python 'lists', which are different than numpy 'array' objects.
# The array object can do more, but lists are sometimes
# easier to work with.
xpts = []
ypts = []

# Read in the data by looping over each line.
tot = 0.0
for line in infile:
    
    # Split the line up by whitespace delimiters and cast all 
    # the entries as floats.
    vals = np.array(line.split()).astype('float')

    xpts.append(vals[0])
    ypts.append(vals[1])

    tot += vals[1]
    print "%f %f" % (vals[0],tot)


# Plot and format!
plt.plot(xpts,ypts,'bo',markersize=2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
#plt.xlim(0,10)

plt.show()
