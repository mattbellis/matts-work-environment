# Routine linear fit demo.
# Adapted from http://physics.nyu.edu/pine/pymanual/graphics/graphics.html
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.optimize

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
def linear(x,m,b):
    return (m*x) + b

################################################################################
# main 
################################################################################
def main():

    infile = open('linear_fit_data_scientific_notation.dat')

    # Create some arrays to store the data.
    xdata = np.array([])
    ydata = np.array([])
    yerr  = np.array([])

    ############################################################################
    # Read in the data.
    ############################################################################
    for line in infile:
        vals = line.split()

        if len(vals)>0:
            xdata = np.append(xdata,float(vals[0]))
            ydata = np.append(ydata,float(vals[1]))
            yerr  = np.append(yerr,float(vals[2]))
    
    ############################################################################

    # Perform linear fit using Levenburg-Marquardt algorithm
    # Take some guesses at what the values are 
    m = 1.1 # slope
    b = 0.4 # y-intercept
    nlfit, nlpcov = scipy.optimize.curve_fit(linear, xdata, ydata, p0=[m,b], sigma=yerr)

    # Unpack output and give outputs of fit nice names
    m_fit, b_fit = nlfit           # returned values of fitting parameters
    dm_fit = np.sqrt(nlpcov[0][0]) # uncertainty in 1st fitting parameter
    db_fit = np.sqrt(nlpcov[1][1]) # uncertainty in 2nd fitting parameter

    # Compute reduced chi square.
    # Need to do this to get proper uncertainties on the fit values.
    rchisq = RedChiSqr(linear, xdata, ydata, yerr, nlfit)

    # Create fitted data for plotting the fit function over the measured data.
    qfit = np.array([xdata.min(), xdata.max()])
    Sfit = linear(qfit, m_fit, b_fit)

    # Plot data and fit
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(qfit, Sfit)

    # Set the x-range on the limits
    #ax1.set_xlim(-1,5.0)

    ax1.errorbar(xdata,ydata,yerr, fmt="or", ecolor="black")

    ax1.set_xlabel("x-axis",fontsize=20)
    ax1.set_ylabel("y-axis",fontsize=20)

    ax1.text(0.7, 0.83,
                 "slope $= {0:0.3e} \pm {1:0.3e}$".format(m_fit, dm_fit),
                      ha="right", va="bottom", transform = ax1.transAxes,fontsize=20)
    ax1.text(0.7, 0.90,
            "intercept $= {0:0.3f} \pm {1:0.3f}$".format(b_fit, db_fit),
                 ha="right", va="bottom", transform = ax1.transAxes,fontsize=20)

    plt.savefig("fitted_data.png")
    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()
