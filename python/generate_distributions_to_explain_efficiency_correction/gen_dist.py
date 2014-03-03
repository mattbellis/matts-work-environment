import lichen.lichen as lch

import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats

################################################################################
# Efficiency
################################################################################
def efficiency(x):

    xeff = []
    for xpt in x:
        eff = (300-xpt)/200.0

        if np.random.random()<eff:
            xeff.append(xpt)

    xeff = np.array(xeff)

    return xeff




pdf = stats.norm(173,10)
x_org = pdf.rvs(1000)
x_data = efficiency(x_org)

mc_raw = 200*np.random.random(100000) + 100
mc_acc = efficiency(mc_raw)

plt.figure()
hraw,xptsraw,yptsraw,xpts_errraw,ypts_errraw = lch.hist_err(mc_raw,bins=50,range=(120,220),linewidth=3,ecolor='b',label='MC truth')
hacc,xptsacc,yptsacc,xpts_erracc,ypts_erracc = lch.hist_err(mc_acc,bins=50,range=(120,220),linewidth=3,ecolor='k',label='MC reconstructed')
plt.xlabel("Mass")
plt.ylim(0)
plt.legend()
plt.savefig('mc.png')

plt.figure()
delta = np.sqrt((yptsacc/yptsraw*(1- yptsacc/yptsraw))/yptsraw)
plt.errorbar(xptsraw, yptsacc/yptsraw, xerr=xpts_errraw, yerr=delta,markersize=2,ecolor='r',fmt='o',linewidth=3,label='Efficency from MC')
plt.xlabel('Mass')
plt.ylabel('Efficiency')
plt.ylim(0,1.0)
plt.legend()
plt.savefig('eff.png')

plt.figure()
hdata,xptsdata,yptsdata,xpts_errdata,ypts_errdata = lch.hist_err(x_data,bins=50,range=(120,220),linewidth=3,ecolor='k',label='Data')
plt.xlabel("Mass")
plt.ylim(0,100)
plt.legend(loc=2)
plt.savefig('data.png')
plt.errorbar(xptsdata, yptsdata/(yptsacc/yptsraw), xerr=xpts_errraw, yerr=np.sqrt(delta*delta+ypts_errdata*ypts_errdata),markersize=2,ecolor='c',fmt='o',linewidth=3,label='Data corrected for efficency')
plt.legend(loc=2)
plt.savefig('data_wcorrected.png')
lch.hist_err(x_org,bins=50,range=(120,220),linewidth=3,ecolor='b',label='Data truth')
plt.legend(loc=2)
plt.savefig('data_wtruth.png')


plt.show()
