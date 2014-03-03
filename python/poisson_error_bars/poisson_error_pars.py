import scipy.stats as stats
import numpy as np

def continuous_poisson(obs,mean):

    ret = (mean**obs)*np.exp(-mean)/stats.gamma.pdf(mean,obs+1)

    return ret


mean = 2

lo = mean - np.sqrt(mean)
hi = mean + np.sqrt(mean)

print "%f + %f - %f" % (mean,lo,hi)

#print stats.poisson.cdf(4,mean)

width = 0.1
if mean<=5:
    width = 0.005

max_hi = mean+(1.5*np.sqrt(mean))


found_lo = False
found_hi = False

i = mean-(1.5*np.sqrt(mean))
#'''
if i<0:
    i=0
#'''

while i<max_hi:
    val = stats.poisson.cdf(i,mean)

    if not found_lo:
        if val<0.16:
            lo = i
        else:
            found_lo = True

    if not found_hi:
        if val<0.84:
            hi = i
        else:
            found_hi = True

    i+=width

print "%f + %f - %f" % (mean,lo,hi)

################################################################################
# Continuous poisson
################################################################################

found_lo = False
found_hi = False

i = mean-(1.5*np.sqrt(mean))
#'''
if i<0:
    i=0
#'''

tot = 0
while i<max_hi:
    val = continuous_poisson(i,mean)
    print val

    tot += val

    if not found_lo:
        if tot<0.16:
            lo = i
        else:
            found_lo = True

    if not found_hi:
        if tot<0.84:
            hi = i
        else:
            found_hi = True

    i+=width

print "%f + %f - %f" % (mean,lo,hi)
