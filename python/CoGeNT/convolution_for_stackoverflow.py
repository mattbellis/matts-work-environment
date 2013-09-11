import numpy as np
import matplotlib.pylab as plt

import scipy.signal as signal
import scipy.stats as stats

# Create the initial function. I model a spike
# as an arbitrarily narrow Gaussian
mu = 1.0 # Centroid
sig=0.2 # Width
original_pdf = stats.norm(mu,sig)

x = np.linspace(0.0,2.0,1000)
y = original_pdf.pdf(x)
plt.plot(x,y/y.sum(),label='original')


# Create the ``smearing" function to convolve with the 
# original function.
# I use a Gaussian, centered at 0.0 (no bias) and 
# width of 0.5
mu_conv = 0.0 # Centroid 
sigma_conv = 0.1 # Width
convolving_term = stats.norm(mu_conv,sigma_conv)
#convolving_term = stats.norm(mu_conv,lambda x: 0.2*x + 0.1)

xconv = np.linspace(-5,5,5000)
yconv = convolving_term.pdf(xconv)

convolved_pdf = signal.convolve(y/y.sum(),yconv,mode='same')

plt.plot(x,convolved_pdf/convolved_pdf.sum(),label='convolved')
#plt.ylim(0,1.2*max(convolved_pdf))
plt.legend()

plt.figure()
plt.plot(x,(convolved_pdf/convolved_pdf.sum())/(y/y.sum()),label='convolved')

plt.show()
