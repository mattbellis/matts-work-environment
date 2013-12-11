import numpy as np
import matplotlib.pylab as plt

import sys

################################################################################
# Read in one of the microprobe data files and return the x and y points, scaled where
# x=0 is equal to y=0.4 (fractional).
################################################################################
def read_in_a_microprobe_data_file(infilename):

    infile = open(infilename)

    x = []
    y = []

    x_closest_to_40 = -999
    min_diff = 1e6
    for i,line in enumerate(infile):

        if i!=0:

            vals = line.split(',')

            concentration = float(vals[1])
            xpos = float(vals[5])

            difference = abs(concentration-40.0)
            if min_diff>difference:
                min_diff = difference
                x_closest_to_40 = i

            y.append(concentration)
            x.append(xpos)


    print x
    print x_closest_to_40,x[x_closest_to_40]

    y = np.array(y)
    y /= 100 # Convert from % to fraction

    x = np.array(x)
    # Shift the points so that the 0 position is close to the 40% mark.
    x -= x[x_closest_to_40]

    x /= 1000000.0 # Convert to meters

    print x[0],x[-1]


    return x,y

################################################################################
# Read in one of the data files and return the x and y points, scaled where
# x=0 is equal to y=0.4 (fractional).
################################################################################
def read_in_an_isotope_data_file(infilename):

    infile = open(infilename)

    x = []
    y = []
    yerr = []
    c = []

    for i,line in enumerate(infile):

        if i!=0:

            vals = line.split(',')

            xpos = float(vals[3])
            delta = float(vals[6])
            deltaerr = float(vals[7].split()[1])
            Fe_conc = float(vals[4])

            x.append(xpos)
            y.append(delta)
            yerr.append(deltaerr)
            c.append(Fe_conc)


    y = np.array(y)
    yerr = np.array(yerr)

    x = np.array(x)
    x /= 1000000.0 # Convert microns to meters.

    c = np.array(c)
    c /= 100.0 # Convert % to fraction.

    return x,y,yerr,c





################################################################################
################################################################################
if __name__=="__main__":

    x,y = read_in_a_microprobe_data_file(sys.argv[1])

    plt.plot(x,y)
    plt.ylim(0.0,1.1)
    plt.show()