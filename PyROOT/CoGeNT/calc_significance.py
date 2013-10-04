import sys
import numpy as np
import scipy.stats as stats

lh0 = float(sys.argv[1])
lh1 = float(sys.argv[2])

delta_ndof = float(sys.argv[3])

D = 2*np.abs(lh0 - lh1)

sig = stats.chisqprob(D,delta_ndof)

print "\n\n"
print "D:   %f" % (D)
print "sig: %f" % (sig)
