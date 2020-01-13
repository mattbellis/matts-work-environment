import matplotlib.pylab as plt
import numpy as np

import scipy.stats as stats
import lichen as lch


################################################################################
# Basic example with histograms
################################################################################
'''
x = np.linspace(100,750,1000)

pdf1 = stats.expon(loc=100,scale=200.0)
bkg = pdf1.rvs(size=1000000)

pdf2 = stats.norm(loc=400,scale=20.0)
sig = pdf2.rvs(size=1000000)

plt.figure(figsize=(4,6))
plt.subplot(2,1,1)
#plt.hist([bkg,sig],bins=100,range=(100,750),stacked=True)
#plt.hist(bkg,bins=100,range=(100,750),label='MC background')
plt.plot(x,pdf1.pdf(x),linewidth=4,label='MC background')
plt.xlim(50,800)
plt.ylabel('Entries')
plt.xlabel(r'Invariant mass [GeV/c$^2$]')
plt.legend()

plt.subplot(2,1,2)
#plt.hist(sig,bins=100,range=(100,750),color='red',label='MC signal')
plt.plot(x,pdf2.pdf(x),linewidth=4,color='red',label='MC signal')
plt.xlim(50,800)
plt.ylabel('Entries')
plt.xlabel(r'Invariant mass [GeV/c$^2$]')
plt.legend()
plt.tight_layout()
plt.savefig('example1_MC.png')

data = bkg[0:10000].tolist() + sig[0:1000].tolist()

plt.figure()
lch.hist(data,bins=100,range=(100,750),label='Data')
plt.ylabel('Entries',fontsize=18)
plt.xlabel(r'Invariant mass [GeV/c$^2$]',fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig('example1_data.png')
'''
################################################################################


################################################################################
# Basic example
################################################################################
'''

pdf1 = stats.weibull_min(1.79,loc=1,scale=1.0)
bkg1 = pdf1.rvs(size=1000000)

pdf2 = stats.weibull_max(2.87,loc=3.5,scale=1.0)
bkg2 = pdf2.rvs(size=500000)

pdf3 = stats.norm(loc=2.0,scale=0.15)
sig = pdf3.rvs(size=50000)

plt.figure(figsize=(5,4))
plt.hist(bkg1,bins=100,range=(1,4),label='MC background 1',color='#ff7f03')
plt.ylabel('Entries')
plt.xlabel(r'ML variable')
plt.legend()
plt.tight_layout()
plt.savefig('example2_MC1.png')

plt.figure(figsize=(5,4))
plt.hist(bkg2,bins=100,range=(1,4),label='MC background 2',color='#2ca02c')
plt.ylabel('Entries')
plt.xlabel(r'ML variable')
plt.legend()
plt.tight_layout()
plt.savefig('example2_MC2.png')

plt.figure(figsize=(5,4))
plt.hist(sig,bins=100,range=(1,4),label='MC signal',color='#d62728')
plt.ylabel('Entries')
plt.xlabel(r'ML variable')
plt.legend()
plt.tight_layout()
plt.savefig('example2_MC3.png')

#plt.subplot(2,1,2)
##plt.hist(sig,bins=100,range=(100,750),color='red',label='MC signal')
#plt.plot(x,pdf2.pdf(x),linewidth=4,color='red',label='MC signal')
#plt.xlim(50,800)
#plt.ylabel('Entries')
#plt.xlabel(r'Invariant mass [GeV/c$^2$]')
#plt.legend()
#plt.tight_layout()
#plt.savefig('example1_MC.png')
#
data = bkg1[0:10000].tolist() 
data += bkg2[0:5000].tolist() 
data += sig[0:500].tolist()

for i in [0,1]:
    plt.figure(figsize=(5,4))
    lch.hist(data,bins=100,range=(1,4),label='Data',color='k')
    if i==1:
        plt.hist([bkg1,bkg2,sig],weights=[0.01*np.ones_like(bkg1), 0.01*np.ones_like(bkg2),0.01*np.ones_like(sig)]  ,stacked=True,bins=100,range=(1,4),label=['MC background 1','MC background 2','MC signal'])
    plt.ylabel('Entries',fontsize=18)
    plt.xlabel(r'ML variable',fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig('example2_data_{0}.png'.format(str(i)))
'''
################################################################################
# Breakdown of Higgs single bin stuff
################################################################################
#'''
data = [125,25,5.5,5]
zz = [125,6.8,5.5,0.3]
zx = [125,2.6,5.5,0.4]
higgs125 = [125,17.3,5.5,1.3]
higgs126 = [125,19.6,5.5,1.5]

plt.figure(figsize=(5,4))
#plt.errorbar(data[0],data[1],fmt='ko',xerr=data[2],yerr=data[3])
plt.errorbar(data[0],data[1],fmt='ko',xerr=data[2],yerr=0,markersize=10)
#plt.errorbar(zz[0],zz[1],fmt='o',xerr=zz[2],yerr=zz[3],markersize=10)
#plt.errorbar(zx[0],zx[1],fmt='o',xerr=zx[2],yerr=zx[3],markersize=10)
plt.xlabel(r'M(4$\ell$) (GeV)',fontsize=18)
plt.ylabel(r'Events',fontsize=18)
plt.ylim(0,1.5*data[1])
plt.xlim(100,150)
plt.tight_layout()
plt.savefig('higgs_example_data.png')

plt.figure(figsize=(5,4))
plt.errorbar(data[0],data[1],fmt='ko',xerr=data[2],yerr=0,markersize=10)
plt.hist([[zz[0]],[zx[0]],[higgs125[0]]],weights=[[zz[1]],[zx[1]],[higgs125[1]]],bins=1,range=(data[0]-data[2],data[0]+data[2]),stacked=True)#,color=['r','b'])
#plt.errorbar(zz[0],zz[1],fmt='o',xerr=zz[2],yerr=zz[3],markersize=10)
#plt.errorbar(zx[0],zx[1],fmt='o',xerr=zx[2],yerr=zx[3],markersize=10)
# Error smear
llx = data[0]-data[2] # lower left,x
lly = data[1]-1.5 # lower left,y
width = 2*data[2] 
height = 2*1.5
rect = plt.Rectangle([llx,lly],width,height,fc='k',alpha=0.5,hatch='x')
plt.gca().add_patch(rect)

plt.xlabel(r'M(4$\ell$) (GeV)',fontsize=18)
plt.ylabel(r'Events',fontsize=18)
plt.ylim(0,1.5*data[1])
plt.xlim(100,150)
plt.tight_layout()
plt.savefig('higgs_example_data_stacked.png')


i = 0
for val,color in zip([zz,zx,higgs125],['#1f77b4','#ff7f03','#2ca02c']):
    plt.figure(figsize=(5,4))
    
    plt.hist([val[0]],weights=[val[1]],bins=1,range=(data[0]-data[2],data[0]+data[2]),color=color)
    llx = val[0]-val[2] # lower left,x
    lly = val[1]-val[3] # lower left,y
    width = 2*val[2] 
    height = 2*val[3]
    rect = plt.Rectangle([llx,lly],width,height,fc='k',alpha=0.5,hatch='/')
    plt.gca().add_patch(rect)
    plt.ylim(0,1.2*val[1])
    plt.xlim(100,150)
    plt.xlabel(r'M(4$\ell$) (GeV)',fontsize=18)
    plt.ylabel(r'Events',fontsize=18)
    plt.tight_layout()
    plt.savefig('higgs_example_data_{0}.png'.format(str(i)))
    i += 1
#'''


################################################################################
# Uncertainty distributions
################################################################################

x = np.linspace(0.5,8,1000)

pdf1 = stats.norm(loc=2.6,scale=0.4)
# Mode, peak position is exp(mu-sigma^2)
# https://en.wikipedia.org/wiki/Log-normal_distribution
#pdf2 = stats.lognorm(s=0.4,loc=1.115,scale=np.exp(1.115))#,scale=np.exp(2.6))
# Just trying to do it by hand
pdf2 = stats.lognorm(s=0.4,loc=1.70,scale=1.1)

plt.figure(figsize=(6,4))
plt.plot([2.6,2.6],[0,1.5],'k--',label='Central value prediction')
plt.plot(x,pdf1.pdf(x),'-.',linewidth=4,label='Normal distribution')
plt.plot(x,pdf2.pdf(x),linewidth=4,label='Log-normal distribution')
plt.xlim(0,8)
plt.xlabel(r'# of events from $Z$ + $X$ background prediction',fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('normal_lognormal.png'.format(str(i)))


plt.show()
