import matplotlib.pylab as plt
import numpy as np

# Define your file name.
infilename = 'example_data.dat'
infile = open(infilename,'r')

# Holders for your x and y points. Note that these are just
# python 'lists', which are different than numpy 'array' objects.
# The array object can do more, but lists are sometimes
# easier to work with.
xpts = []
ypts = []

# Read in the data by looping over each line.
for line in infile:
    
    # Split the line up by whitespace delimiters and cast all 
    # the entries as floats.
    vals = np.array(line.split()).astype('float')

    xpts.append(vals[0])
    ypts.append(vals[1])


# Plot and format!
plt.plot(xpts,ypts,'bo',markersize=20)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(0,10)

plt.show()
