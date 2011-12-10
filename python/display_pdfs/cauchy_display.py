import numpy as np
import pylab as plt
from scipy.stats import cauchy

#### dweibull
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dweibull.html#scipy.stats.dweibull
numargs = cauchy.numargs
[ ] = [8.0,] * numargs
rv = cauchy()

x = np.linspace(0, np.minimum(rv.dist.b, 3))
h = plt.plot(x, rv.pdf(x))

# Check accuracy of cdf and ppf
#prb = cauchy.cdf(x, c)
#h = plt.semilogy(np.abs(x - cauchy.ppf(prb, c)) + 1e-20)

# Random number generation
#R = cauchy.rvs(c, size=100)

plt.show()
