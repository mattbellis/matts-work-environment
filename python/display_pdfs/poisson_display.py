import numpy as np
import pylab as plt
from scipy.stats import poisson

#### poisson
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson
numargs = poisson.numargs
[ mu ] = [3,] * numargs
rv = poisson(mu)

x = np.arange(0, np.minimum(rv.dist.b, 10)+1)
# Must use pmf instead of pdf.
print rv.pmf(x)
h = plt.plot(x, rv.pmf(x))

# Check accuracy of cdf and ppf
#prb = poisson.cdf(x, mu)
#h = plt.semilogy(np.abs(x - poisson.ppf(prb, mu)) + 1e-20)

# Random number generation
#R = poisson.rvs(c, size=100)

plt.show()
