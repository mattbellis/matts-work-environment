import matplotlib.pylab as plt
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

################################################################################
# BNV
################################################################################
y = [180,520,6.2,8.1,6.1,3.2]
yerr = [0.005, 0.005, 0.15e-3, 1.4e-4, 0.14e-4, 0.35e-4]

y = np.array(y)
yerr = np.array(yerr)

y *= 1e-8
yerr *= 1e-8

x = np.arange(0,len(y))
print x
mylabels = [r'$\Lambda_c^+ \mu^-$',\
            r'$\Lambda_c^+ e^-$', \
            r'$\Lambda^0 \mu^-$', \
            r'$\Lambda^0 e^-$', \
            r'$\bar{\Lambda}^0 \mu^-$', \
            r'$\bar{\Lambda}^0 e^-$'
            ]

################################################################################
# LNV (same flavour)
################################################################################
ylnv = [2.3,3.0,10.7,6.7]
ylnverr = [0.005, 0.005, 0.15e-3, 1.4e-4]

ylnv = np.array(ylnv)
ylnverr = np.array(ylnverr)

ylnv *= 1e-8
ylnverr *= 1e-8

xlnv = np.arange(len(x),len(ylnv)+len(x))
mylabelslnv = [r'$\pi^- e^+ e^+$',\
               r'$K^- e^+ e^+$',\
               r'$\pi^- \mu^+ \mu^+$',\
               r'$K^- \mu^+ \mu^+$'
            ]


boxes = []
patches = []

################################################################################
# LNV (same and opposite flavour)
################################################################################
ylnv2 = [4.0,3.0,5.9,1.7,4.7,4.2,26,21,17,1.6,1.5]
xlnv2 = np.arange(len(x),len(ylnv2)+len(x))

ylnv2 = np.array(ylnv2)
#ylnv2err = np.array(ylnv2err)

ylnv2 *= 1e-7
#ylnv2err *= 1e-7

yerrlnv2lo = 1e-8*np.ones(len(ylnv2))
yerrlnv2hi =     np.zeros(len(ylnv2))

text_offsetslnv2 = 0.4*np.ones(len(ylnv2))
xlnv2 = np.arange(len(x)+len(xlnv),len(ylnv2)+len(x)+len(xlnv))

mylabelslnv2 = [r'$K^{*-} e^+ e^+$',\
                r'$K^{*-} e^+ \mu^+$',\
                r'$K^{*-} \mu^+ \mu^+$',\
                r'$\rho^{-} e^+ e^+$',\
                r'$\rho^{-} e^+ \mu^+$',\
                r'$\rho^{-} \mu^+ \mu^+$',\
                r'$D^{-} e^+ e^+$',\
                r'$D^{-} e^+ \mu^+$',\
                r'$D^{-} \mu^+ \mu^+$',\
                r'$K^{-} e^+ \mu^+$',\
                r'$\pi^{-} e^+ \mu^+$'
               ]

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
plt.errorbar(x,y,yerr=yerr,fmt='o',markersize=20,label=r'BNV/LNV decays (BaBar) 2011')

print xlnv
print ylnv
print ylnverr
plt.errorbar(xlnv,ylnv,yerr=ylnverr,fmt='s',color='green',markersize=20,linewidth=5,label=r'LNV same flavour decays (BaBar) 2012')

print yerrlnv2lo
print yerrlnv2hi
plt.errorbar(xlnv2,ylnv2,yerr=[yerrlnv2lo,yerrlnv2hi],fmt='^',color='red',markersize=20,linewidth=5,label=r'LNV same/different flavour decays (BaBar) 2013')
#ylnv2 = np.array(ylnv2)


print mylabels
print mylabelslnv
print mylabelslnv2
ticks = np.append(x,xlnv)
ticks = np.append(ticks,xlnv2)
plt.xticks(ticks,mylabels+mylabelslnv+mylabelslnv2,fontsize=18)

plt.xlim(min(x)-1,max(xlnv2)+1)
plt.ylim(1e-8,2e-4)
plt.yscale('log')
plt.ylabel(r'$\mathcal{B}_{\rm UL}$ at 90% CL',fontsize=24)

plt.plot([5.5,5.5],[1e-8,1e-3],'k--')
plt.plot([9.5,9.5],[1e-8,1e-3],'k--')

plt.text(0.15,0.65,r'$\Lambda_{(c)} \ell^\pm$',transform=plt.gca().transAxes,fontsize=32,color='blue')
plt.text(0.31,0.55,r'$h^\mp \ell^\pm \ell^\pm$',transform=plt.gca().transAxes,fontsize=32,color='green')
plt.text(0.52,0.60,r'$h^\mp \ell^\pm \ell^{\'\pm}$',transform=plt.gca().transAxes,fontsize=32,color='red')

locs, labels = plt.xticks()
plt.setp(labels, rotation=-60)
plt.subplots_adjust(bottom=0.22,right=0.98,left=0.09,top=0.98)

plt.legend()

plt.savefig('BaBar_BNV_LNV.png')

plt.show()


