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
def read_datafile(infilename,experiment="FNDA1",element="Fe",profile="1"):

    vals = np.loadtxt(infilename)
    vals = vals.transpose()
    spot_number,posx,posy,dfe,conc_fe,conc_ni,delta_fe,delta_fe_err = vals[0:8]
    # dfe = distance from edge

    print(dfe)
    print(conc_fe)
    print(conc_ni)

    crossing_point = 40.0 # For Fe
    if element=="Fe":
        crossing_point = 40.0 # For Fe
    elif element=="Ni":
        crossing_point = 60.0 # For Ni

    print("Element %s" % (element))

    # Flip the Ni x vals
    if element=="Ni":
        #dfe = dfe[::-1]
        dfe = -dfe

    if experiment=="FNDA1" and element=="Fe" and profile=="2":
        dfe = -dfe

    # Find the x (dfe) closest to the crossing point
    predicted_conc = interpolate.interp1d(dfe,conc_fe)
    if element=="Ni":
        predicted_conc = interpolate.interp1d(dfe,conc_ni)

    xvals = np.linspace(min(dfe),max(dfe),1000)
    idx = find_nearest(predicted_conc(xvals),crossing_point)
    xoffset = xvals[idx]
    print("OFFSET: %f" % (xoffset))
    print("xoffset: %f %e" % (xoffset,xoffset))
    if element=="Ni":
        dfe -= xoffset
    else:
        dfe -= xoffset

    #plt.plot(xvals-xoffset,predicted_conc(xvals))
    #plt.plot(dfe,conc_ni,'o')
    #dfe -= 2050.

    #conc_ni = 100.0 - conc_ni

    dfe /= 1e6 # Convert to meters
    
    delta_fe *= 2
    #delta_fe_err *= 2

    index = None
    #if 1:
    '''
    if experiment=="FNDA1" and element=="Fe" and profile=="2":
        index =  np.arange(0,10,1)
        index0 = np.arange(15,len(dfe),1)
        index = np.append(index,index0)
        dfe = dfe[index]
        conc_fe = conc_fe[index]
        conc_ni = conc_ni[index]
        delta_fe = delta_fe[index]
        delta_fe_err = delta_fe_err[index]
    '''

    '''
    if experiment=="FNDA1" and element=="Ni" and profile=="2":
        dfe -= 1.3e-4
    '''

    '''
    if experiment=="FNDA2" and element=="Ni" and profile=="1":
        dfe -= 1.3e-4
    '''



    #print index
    print(dfe)
    #exit()
    return dfe,conc_fe/100.,conc_ni/100.,delta_fe,delta_fe_err,xoffset


################################################################################
################################################################################
if __name__=="__main__":

    x,c0,c1,delta,delta_err,xoffset = read_datafile(sys.argv[1],element="Fe",experiment="FNDA1",profile='2')

    plt.figure(figsize=(8,6))
    plt.plot(x,c0,'o',label='Fe')
    plt.plot(x,c1,'o',label='Ni')
    plt.plot([0.0,0.0],[0,1])
    plt.plot([min(x),max(x)],[0.6,0.6])
    plt.plot([min(x),max(x)],[0.4,0.4])
    print(x)
    print(c0)
    plt.ylim(0.0,1.1)
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.errorbar(x,delta,yerr=delta_err,fmt='o')

    plt.show()

################################################################################


