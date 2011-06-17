import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

max = 10000

frac_sig = 0.30


outfile = open('default.txt', 'w+')

xmin, xmax = 0.05, 0.20

# Define the 2D plane
amin, amax = -0.5, 0.5
bmin, bmax = -3.0, 3.0

ar = amax-amin
br = bmax-bmin
xr = xmax-xmin

xarr = []
aarr = []
barr = []

for i in range(0, max):
  if i%100==0:
    print i

  x = 0.0
  a = amin + ar*np.random.rand()
  b = bmin + br*np.random.rand()
  aarr.append(a)
  barr.append(b)

  # Check if signal or background
  test = np.random.rand()
  #print test
  if test<frac_sig: # signal
    mu, sigma = 0.134, 0.010
    x = mu + sigma * np.random.randn()
    xarr.append(x)

  else: # background
    # Bkg shape = polynomial
    slope, intercept = -0.4, 0.10
    prob_max = xmin*slope + intercept
    #print prob_max

    while (1):
      x_test = xmin + xr*np.random.rand()
      prob_test = prob_max*np.random.rand()
      y = intercept + slope*x_test

      #print "%f %f %f" % (x_test, y, prob_test)
      if prob_test < y:
        x = x_test
        xarr.append(x)
        break

  output = "%.4f %.4f %.4f \n" % (x, a, b)
  #print output
  outfile.write(output)

outfile.close()

# Prepare a place to draw the histo
fig = plt.figure()
ax = fig.add_subplot(111)

print len(xarr)
print len(aarr)
print len(barr)

# the histogram of the data
n, bins, patches = ax.hist(xarr, 100, range=(xmin,xmax), facecolor='green', alpha=0.75, histtype='stepfilled')

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
y = mlab.normpdf( bincenters, mu, sigma)
l = ax.plot(bincenters, y, 'r--', linewidth=1)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.grid(True)

plt.show()



