#!/usr/bin/env python

import sys

normalized_val = float(sys.argv[1])
nbins = int(sys.argv[2])

# bins number 1 to nbins
# 0 is the underflow
# nbins+1 is the overflow
# AM I MISSING OUT ON IF VAL==1?????
# Remember that we have underflow (0) and overflow (nbins-1)
if (normalized_val==1.0):
    print nbins-2;
elif (normalized_val>1.0):
    # If it is greater 1.0, put it in the overflow bin
    print nbins-1;
elif (normalized_val<0.0):
    print 0;
elif (normalized_val==0.0):
    print 1;

# Do this calculation only if it fails the other conditionals.
# I think this buys us some CPU cycles.
ret = (int)(normalized_val*(nbins)) + 1;
print ret;

