import numpy as np
import matplotlib.pylab as plt

import sys

sigf = np.load(sys.argv[1])
bkgf = np.load(sys.argv[2])

nsig = sigf[1]
nbkg = bkgf[1]

nsig_surv = sigf[2]
nbkg_surv = bkgf[2]

sig_eff = (nsig_surv/nsig)
bkg_rej = 1.0 - (nbkg_surv/nbkg)

print sig_eff


plt.figure()
print len(bkg_rej),len(sig_eff)
plt.plot(bkg_rej,sig_eff,'o')
plt.xlabel('Background rejection (fractional)')
plt.ylabel('Signal efficiency (fractional)')

plt.figure()
#fom_cutoff = 0.324 # For a=3
#fom_cutoff = 0.194 # For a=5
#for a,fmt in zip([3,4,5],['bo','go','ro']):
for a,fmt in zip([5],['ro']):
    bkg_scale_factor = 1.0
    fom = sig_eff/(np.sqrt(bkg_scale_factor*nbkg_surv) + a/2.0)
    fom_sorted = np.array(fom)
    fom_sorted.sort()
    print "here"
    print fom_sorted
    fom_cutoff = fom_sorted[-6]
    print "fom_cutoff: ",fom_cutoff
    plt.plot(fom,sig_eff,fmt)
    #plt.plot(fom,nbkg_surv,'o')
    print " ---------- EFFECTS -------------"
    if a==5:
        print "fom       : ",fom[fom>fom_cutoff]
        print "sig       : ",sig_eff[fom>fom_cutoff]
        print "bkg-ref   : ",bkg_rej[fom>fom_cutoff]
        br = bkg_rej[fom>fom_cutoff]
        print "bkg remain: ",(1.0-br)*nbkg[fom>fom_cutoff]
        print " ---------- cuts -------------"
        for i in xrange(7):
            print sigf[0][i][fom>fom_cutoff]

#title = "Punzi figure of merit (a=%d)" % (a)
title = "Punzi figure of merit"
plt.xlabel(title)
plt.ylabel('Signal efficiency (fractional)')

plt.show()
