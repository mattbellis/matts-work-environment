import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate

import scipy.signal as signal
import scipy.stats as stats

npts = 200

# Create the initial function. I model a spike
# as an arbitrarily narrow Gaussian

mu  = 1.0 # Centroid
sig = 0.2 # Width
original_pdf = stats.norm(mu,sig)

# Here's the original function as x,y points
x = np.linspace(0,2.0,npts)
y = original_pdf.pdf(x)

plt.plot(x,y,label='original',linewidth=3)

# Create the ``smearing" function to convolve with the 
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and 
# variable width.

mu_conv = 0.0 # Centroid 
tot_conv_pdf = np.zeros(len(y))

for s in range(0,npts):
    #sigma_conv = 0.2 + (0.2*x[s]*x[s]) # Width
    sigma_conv = 0.02 + (0.1*x[s]) # Width
    #sigma_conv = 0.2 # Width
    #sigma_conv = 0.1
    convolving_term = stats.norm(mu_conv,sigma_conv)

    xconv = np.linspace(-5,5,5*npts)
    # Use a normalized Gaussian.
    yconv = convolving_term.pdf(xconv)

    # Multiply by sigma because we want the height constant, not the area.
    # This gives incorrect results
    #yconv = sigma_conv*(np.sqrt(2*np.pi))*convolving_term.pdf(xconv)

    ytemp = np.zeros(npts)
    ytemp[s] = y[s]

    if ytemp[s]>0:
        # Convolve a single point in the original function.
        convolved_pdf = signal.fftconvolve(ytemp,yconv,mode='same')

        #print s,x[s],y[s],sigma_conv,convolved_pdf[500]
        # Sum up each of the contributions.
        tot_conv_pdf += convolved_pdf

# Normalize the final function. This assumes we started
# with a normalized function.
yc = tot_conv_pdf/integrate.simps(tot_conv_pdf,x=x)

plt.plot(x,yc,label='convolved',linewidth=3)
plt.legend()

print "Integral of original function:  %f" % (integrate.simps(y,x=x))
print "Integral of convolved function: %f" % (integrate.simps(yc,x=x))


plt.show()
