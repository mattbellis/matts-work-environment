import numpy as np
import pylab as plt
from scipy.stats import dweibull

#### dweibull
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dweibull.html#scipy.stats.dweibull
numargs = dweibull.numargs
[ c ] = [8.0,] * numargs
rv = dweibull(c,loc=0.001,scale=5.0)

x = np.linspace(0, np.minimum(rv.dist.b, 6))
h = plt.plot(x, rv.pdf(x))

# Check accuracy of cdf and ppf
#prb = dweibull.cdf(x, c)
#h = plt.semilogy(np.abs(x - dweibull.ppf(prb, c)) + 1e-20)

# Random number generation
#R = dweibull.rvs(c, size=100)

plt.show()
