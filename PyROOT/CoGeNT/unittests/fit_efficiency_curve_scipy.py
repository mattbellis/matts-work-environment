# Routine linear fit demo.
# Adapted from http://physics.nyu.edu/pine/pymanual/graphics/graphics.html
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.optimize
import datetime
import sys

################################################################################
# define function to calculate reduced chi-squared
################################################################################
def RedChiSqr(func, x, y, dy, params):
    resids = y - func(x, *params)
    chisq = ((resids/dy)**2).sum()
    return chisq/float(x.size-params.size)

################################################################################
# Define fitting function for linear fit.
################################################################################
def sigmoid(x,Ethresh,sigma):
    max_eff = 0.86786
    sigmoid = max_eff/(1+np.exp((-(x-Ethresh)/(sigma*Ethresh))))
    return sigmoid

################################################################################
# main 
################################################################################
def main():

    ############################################################################
    # Edit these for your particular dataset.
    ############################################################################
    infile = open(sys.argv[1])
    do_fit = True # Can be True or False

    # Create some arrays to store the data.
    xdata = np.array([])
    ydata = np.array([])
    yerr  = np.array([])
    xerr  = np.array([])

    ############################################################################
    # Read in the data.
    ############################################################################
    

    line_count = 0

    starting_time = None
    time_start = None

    for line in infile:
        vals = line.split()
    
        xdata = np.append(xdata,float(vals[0]))
        ydata = np.append(ydata,float(vals[1]))
        xerr = np.append(xerr,0.05)
        yerr = np.append(yerr,0.005)

        line_count += 1

    # Define the y-error to be the square root of the number of counts.

    ############################################################################

    # Plot data and fit
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Set the x-range on the limits
    #ax1.set_xlim(-1,5.0)

    ax1.errorbar(xdata,ydata,yerr,xerr=xerr,fmt="or",ecolor="black")

    ax1.set_xlabel("x-axis",fontsize=20)
    ax1.set_ylabel("y-axis",fontsize=20)

    if do_fit:
        # Perform sigmoid fit using Levenburg-Marquardt algorithm
        # Take some guesses at what the values are 
        #tau = 0.87 # slope
        A = 0.55 # y-intercept
        offset = 0.3 # y-intercept
        nlfit, nlpcov = scipy.optimize.curve_fit(sigmoid, xdata, ydata, p0=[A,offset], sigma=yerr)

        # Unpack output and give outputs of fit nice names
        A_fit,offset_fit = nlfit           # returned values of fitting parameters
        #tau_fit,A_fit,offset_fit = nlfit           # returned values of fitting parameters
        #print tau_fit
        print A_fit
        print offset_fit
        #dtau_fit = np.sqrt(nlpcov[0][0]) # uncertainty in 1st fitting parameter
        dA_fit = np.sqrt(nlpcov[0][0]) # uncertainty in 2nd fitting parameter
        doffset_fit = np.sqrt(nlpcov[1][1]) # uncertainty in 2nd fitting parameter

        # Compute reduced chi square.
        # Need to do this to get proper uncertainties on the fit values.
        rchisq = RedChiSqr(sigmoid, xdata, ydata, yerr, nlfit)

        # Create fitted data for plotting the fit function over the measured data.
        qfit = np.linspace(xdata.min(),xdata.max(),100)
        Sfit = sigmoid(qfit, A_fit, offset_fit)

        ax1.plot(qfit, Sfit)

        '''
        ax1.text(0.7,0.85,r'$y = Ae^{-t/\tau} + k$',fontsize=30,ha="right",va="bottom",transform=ax1.transAxes)
        ax1.text(0.7,0.70,
                     "$\\tau= {0:0.3f} \pm {1:0.3f}$".format(tau_fit, dtau_fit),
                          ha="right", va="bottom", transform = ax1.transAxes,fontsize=20)
        '''

    # Save the image.
    plt.savefig("radiation_lab.png")
    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()
