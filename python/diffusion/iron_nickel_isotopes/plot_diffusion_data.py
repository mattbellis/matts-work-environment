import numpy as np
import matplotlib.pylab as plt

import sys

from scipy import interpolate


################################################################################
# Read in one of the microprobe data files and return the x and y points, scaled where
# x=0 is equal to y=0.4 (fractional).
################################################################################
def read_in_a_microprobe_data_file(infilename,experiment="FNDA1",element="Fe"):

    infile = open(infilename)
    
    crossing_point = 43.0 # For Fe
    if element=="Fe":
        crossing_point = 43.0 # For Fe
    elif element=="Ni":
        crossing_point = 58.0 # For Ni

    x = []
    y = []

    x_closest_to_crossing_point = -999
    min_diff = 1e6
    concentration = None
    for i,line in enumerate(infile):

        if i!=0:

            vals = line.split(',')

            # The ``new files"
            if element=="Fe":
                concentration = float(vals[0]) # Fe
            elif element=="Ni":
                concentration = float(vals[1]) # Ni

            xpos = float(vals[4])

            difference = abs(concentration-crossing_point)
            if min_diff>difference:
                min_diff = difference
                x_closest_to_crossing_point = i-1 # Because we skip the first line.

            y.append(concentration)
            x.append(xpos)

    print x
    print x_closest_to_crossing_point,x[x_closest_to_crossing_point]

    y = np.array(y)
    y /= 100.0 # Convert from % to fraction

    x = np.array(x)
    # Shift the points so that the 0 position is close to the crossing_point% mark.
    x -= x[x_closest_to_crossing_point]

    x /= 1000000.0 # Convert to meters

    if experiment=="FNDA2":
        x *= -1

    print x[0],x[-1]

    return x,y

################################################################################
# Read in one of the data files and return the x and y points, scaled where
# x=0 is equal to y=0.4 (fractional).
################################################################################
def read_in_an_isotope_data_file(infilename,experiment="FNDA1",element="Fe"):

    infile = open(infilename)

    crossing_point = 43.0 # For Fe
    if element=="Fe":
        crossing_point = 43.0 # For Fe
    elif element=="Ni":
        crossing_point = 58.0 # For Ni

    x = []
    y = []
    yerr = []
    c = []

    xpos=delta=deltaerr=conc = None
    for i,line in enumerate(infile):

        print line
        if line[0] != "#":

            vals = line.split(',')

            # The ``new" files, FNDA 1
            if experiment=="FNDA1":
                xpos = float(vals[0])
                # Fe
                if element=="Fe":
                    delta = float(vals[3])
                    deltaerr = float(vals[4].split()[1])
                    conc = float(vals[1])
                # Ni
                elif element=="Ni":
                    conc = float(vals[2])
                    delta = float(vals[5])
                    deltaerr = float(vals[6])

            # The ``new" files, FNDA 2
            elif experiment=="FNDA2":
                xpos = float(vals[0])
                # Fe
                if element=="Fe":
                    delta = float(vals[3])
                    deltaerr = float(vals[4])
                    conc = float(vals[1])
                # Ni
                elif element=="Ni":
                    conc = float(vals[2])
                    delta = float(vals[6])
                    deltaerr = float(vals[7].split()[1])

            x.append(xpos)
            y.append(delta)
            yerr.append(deltaerr)
            c.append(conc)

    if experiment=="FNDA1":
        x = x[::-1] # Do this for FNDA 1, reverse it.
    
    print x[0],x[-1]
    #exit()
    # Figure out where the crossing point is.
    print x
    print c
    #plt.plot(x,c)
    #plt.show()
    predicted_conc = interpolate.interp1d(x,c)
    if experiment=="FNDA1" or experiment=="FNDA2":
        predicted_conc = interpolate.interp1d(x[::-1],c[::-1])
    min_diff = 1e6
    offset = -999
    for i in np.linspace(x[-1],x[0],1000):
        difference = abs(predicted_conc(i)-crossing_point)
        print i,predicted_conc(i),crossing_point,difference
        if difference<min_diff:
            min_diff = difference
            offset = i

    print offset
    x -= offset
    #exit()

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
