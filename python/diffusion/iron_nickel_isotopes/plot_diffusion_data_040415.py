import matplotlib.pylab as plt
import numpy as np

import sys

from scipy import interpolate

################################################################################
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
################################################################################

################################################################################
# Read in the data file
################################################################################
def read_datafile(infilename,experiment="FNDA1",element="Fe"):

    vals = np.loadtxt(infilename)
    vals = vals.transpose()
    spot_number,posx,posy,dfe,conc_fe,conc_ni,delta_fe,delta_fe_err = vals[0:8]
    # dfe = distance from edge

    print dfe
    print conc_fe
    print conc_ni

    crossing_point = 40.0 # For Fe
    if element=="Fe":
        crossing_point = 40.0 # For Fe
    elif element=="Ni":
        crossing_point = 60.0 # For Ni

    # Find the x (dfe) closest to the crossing point
    predicted_conc = interpolate.interp1d(dfe,conc_fe)

    xvals = np.linspace(min(dfe),max(dfe),1000)
    idx = find_nearest(predicted_conc(xvals),crossing_point)
    xoffset = xvals[idx]
    print "xoffset: %f %e" % (xoffset,xoffset)
    dfe -= xoffset

    #dfe -= 2050.

    #conc_ni = 100.0 - conc_ni

    dfe /= 1e6 # Convert to meters
    
    delta_fe *= 2
    delta_fe_err *= 2

    return dfe,conc_fe/100.,conc_ni/100.,delta_fe,delta_fe_err,xoffset


################################################################################
################################################################################
if __name__=="__main__":

    x,c0,c1,delta,delta_err,xoffset = read_datafile(sys.argv[1])

    plt.figure(figsize=(8,6))
    plt.plot(x,c0,'o',label='Fe')
    plt.plot(x,c1,'o',label='Ni')
    plt.ylim(0.0,1.1)
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.errorbar(x,delta,yerr=delta_err,fmt='o')

    plt.show()

################################################################################


