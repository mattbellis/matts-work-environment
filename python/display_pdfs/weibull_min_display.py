import numpy as np
import pylab as plt
from scipy.stats import weibull_min

#### Weibull_min
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
numargs = weibull_min.numargs
[ c ] = [1.9,] * numargs
rv = weibull_min(c)

x = np.linspace(0, np.minimum(rv.dist.b, 3))
h = plt.plot(x, rv.pdf(x))


# Check accuracy of cdf and ppf
#prb = weibull_min.cdf(x, c)
#h = plt.semilogy(np.abs(x - weibull_min.ppf(prb, c)) + 1e-20)

# Random number generation
# R = weibull_min.rvs(c, size=100)

plt.show()
