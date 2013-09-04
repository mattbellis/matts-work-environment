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

x = np.linspace(0.0,2.0,npts)
y = original_pdf.pdf(x)
plt.plot(x,y/y.sum(),label='original',linewidth=1)


# Create the ``smearing" function to convolve with the 
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and 
# width of 0.5
mu_conv = 0.0 # Centroid 
tot_conv_pdf = np.zeros(len(y))
for s in range(0,1000):
    sigma_conv = 0.01+0.0012*s # Width
    print sigma_conv
    convolving_term = stats.norm(mu_conv,sigma_conv)
    #convolving_term = stats.norm(mu_conv,lambda x: 0.2*x + 0.1)

    xconv = np.linspace(-5,5,npts)
    yconv = convolving_term.pdf(xconv)

    #ytemp = np.hstack([y[:s+1],y[s+1:]])
    ytemp = np.zeros(npts)
    ytemp[s] = y[s]

    #convolved_pdf = signal.fftconvolve(y/y.sum(),yconv,mode='same')
    convolved_pdf = signal.fftconvolve(ytemp,yconv,mode='same')

    tot_conv_pdf += convolved_pdf

plt.plot(x,tot_conv_pdf/tot_conv_pdf.sum(),label='convolved')
#plt.ylim(0,1.2*max(convolved_pdf))
plt.legend()
plt.show()
