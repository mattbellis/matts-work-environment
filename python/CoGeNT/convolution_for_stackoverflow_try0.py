import numpy as np
import matplotlib.pylab as plt

import scipy.signal as signal
import scipy.stats as stats

npts = 1000

# Create the initial function. I model a spike
# as an arbitrarily narrow Gaussian
mu = 1.0 # Centroid
sig=0.2 # Width
original_pdf = stats.norm(mu,sig)

x = np.linspace(0,2.0,npts)
y = original_pdf.pdf(x)
plt.plot(x,y/y.sum(),label='original',linewidth=1)


# Create the ``smearing" function to convolve with the 
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and 
# width of 0.5
mu_conv = 0.0 # Centroid 
tot_conv_pdf = np.zeros(len(y))
#for s in range(0,npts):
#for s in range(100,200):
#for s in [500,900]:
for s in range(0,npts):
    #sigma_conv = 0.2 + (0.2*x[s]*x[s]) # Width
    sigma_conv = 0.02 + (0.1*x[s]) # Width
    #sigma_conv = 0.1
    convolving_term = stats.norm(mu_conv,sigma_conv)

    xconv = np.linspace(-5,5,5*npts)
    # Multiply by sigma because we want the height constant, 
    # not the area.
    #yconv = sigma_conv*convolving_term.pdf(xconv)
    yconv = sigma_conv*convolving_term.pdf(xconv)

    ytemp = np.zeros(npts)
    ytemp[s] = y[s]

    #convolved_pdf = signal.fftconvolve(y/y.sum(),yconv,mode='same')
    convolved_pdf = signal.fftconvolve(ytemp,yconv,mode='same')

    #print s,x[s],y[s],sigma_conv,convolved_pdf[500]
    tot_conv_pdf += convolved_pdf

plt.plot(x,tot_conv_pdf/tot_conv_pdf.sum(),label='convolved')
plt.ylim(0,1.2*max(tot_conv_pdf/tot_conv_pdf.sum()))
plt.legend()

plt.figure()
plt.plot(x,(tot_conv_pdf/tot_conv_pdf.sum())/(y/y.sum()),label='convolved')

plt.show()
