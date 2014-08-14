import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

################################################################################
# Exponential 
# The slope is interpreted a negative
################################################################################
def exp(x,slope,xlo,xhi,num_int_points=1000):

    exp_func = stats.expon(loc=0.0,scale=slope)

    xnorm = np.linspace(xlo,xhi,num_int_points)
    ynorm = exp_func.pdf(xnorm)
    normalization = integrate.simps(ynorm,x=xnorm)
    #normalization = 1.0
    #print "pdfs normalization: ",normalization
    
    y = exp_func.pdf(x)/normalization

    return y



################################################################################
# Gaussian
################################################################################
def gauss(x,mean,sigma,xlo,xhi,num_int_points=1000):

    gauss_func = stats.norm(loc=mean,scale=sigma)

    xnorm = np.linspace(xlo,xhi,num_int_points)
    ynorm = gauss_func.pdf(xnorm)
    normalization = integrate.simps(ynorm,x=xnorm)
    #normalization = 1.0
    #print "pdfs normalization: ",normalization
    
    y = gauss_func.pdf(x)/normalization

    return y


################################################################################
# Polynomial
################################################################################
def poly(x,constants,xlo,xhi,num_int_points=1000):

    npts = len(x)

    poly = np.ones(npts)

    xnorm = np.linspace(xlo,xhi,num_int_points)
    ynorm = np.ones(num_int_points)

    for i,c in enumerate(constants):
        poly += c*np.pow(x,(i+1))
        ynorm += c*np.pow(xnorm,(i+1))

    normalization = integrate.simps(ynorm,x=xnorm)
    #print "pdfs normalization: ",normalization
    
    return poly/normalization




