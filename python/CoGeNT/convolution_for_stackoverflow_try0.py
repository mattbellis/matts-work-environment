import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate

import scipy.signal as signal
import scipy.stats as stats

npts = 200

# Create the initial function. I model a spike
# as an arbitrarily narrow Gaussian

mu = 1.0 # Centroid
sig=0.2 # Width
original_pdf = stats.norm(mu,sig)

x = np.linspace(0,2.0,npts)
y = original_pdf.pdf(x)

print y.sum()

y /= y.sum()
print y.sum()

plt.plot(x,y,label='original',linewidth=3)


# Create the ``smearing" function to convolve with the 
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and 
# width of 0.5
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

    print ytemp[s]

    # Convolve a single point in the original function.
    convolved_pdf = signal.fftconvolve(ytemp,yconv,mode='same')

    #print s,x[s],y[s],sigma_conv,convolved_pdf[500]
    # Sum up each of the contributions.
    tot_conv_pdf += convolved_pdf

yc = tot_conv_pdf/tot_conv_pdf.sum()
plt.plot(x,yc,label='convolved',linewidth=3)
plt.ylim(0,1.2*max(tot_conv_pdf/tot_conv_pdf.sum()))
plt.legend()

#print x
#print y
#print yc
print integrate.simps(y,x=x)
print integrate.simps(yc,x=x)

plt.show()
