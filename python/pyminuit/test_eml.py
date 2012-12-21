import numpy as np
import scipy as sp

import scipy.stats as stats
import scipy.integrate as integrate

import matplotlib.pylab as plt

np.random.seed(100)

pdf_sig0 = stats.norm(loc=5.0,scale=0.5)
#pdf_sig1 = stats.norm(loc=6.0,scale=1.0)
#pdf_sig1 = stats.norm(loc=0.0,scale=3.0)
pdf_sig1 = stats.expon(loc=0.0,scale=3.0)

ndata0 = 1000
data0 = pdf_sig0.rvs(ndata0)

ndata1 = 3000
data1 = pdf_sig1.rvs(ndata1)
data1 = data1[data1> 1.0]
data1 = data1[data1<10.0]

print "ndata1: ",len(data1)

data = np.array(data0)
data = np.append(data,data1)

nmc = 10000
#mc = 20.0*np.random.random(nmc) - 10.0
mc = 9.0*np.random.random(nmc) + 1.0

k0 = 1000
k1 = 2041
#k0 = 2000
#k1 = 4082

#k1 = 3000

xpts = np.linspace(1,10,1000)
ypts = pdf_sig1.pdf(xpts)
signorm1 = integrate.simps(ypts,x=xpts)
print "signorm1: ",signorm1

#ktot=float(k0+k1)
ktot=1.0

ndata = ndata0 + len(data1)

lf_data = np.log(k0/ktot*pdf_sig0.pdf(data) + k1/ktot*pdf_sig1.pdf(data)/signorm1).sum()
norm =          (k0/ktot*pdf_sig0.pdf(mc) +   k1/ktot*pdf_sig1.pdf(mc)/signorm1).sum()/nmc

print "-lf_data: ",-lf_data
print "norm: ",norm
print "norm*nmc: ",norm*nmc
print "norm/ndata: ",norm/ndata
print "norm*ndata/nmc: ",norm*ndata/nmc

print "-lf_data + ndata*norm ",-lf_data+(ndata*norm)

norm0 = (k0/ktot*pdf_sig0.pdf(mc)).sum()/nmc
norm1 = (k1/ktot*pdf_sig1.pdf(mc)/signorm1).sum()/nmc

print "ndata: ",ndata
print "fracs: ",ndata*norm0/norm,ndata*norm1/norm

plt.hist(data,bins=50)
#plt.show()

################################################################################
mc = 10.0*np.random.random(5000000)
data = np.random.normal(loc=5.0,scale=0.5,size=1000)

norm = 1.0/np.sqrt(2*np.pi*0.5*0.5)
print "norm: ",norm

ydata = np.exp(-(data-5.0)*(data-5.0)/(2.0*0.5*0.5))
ymc   = np.exp(-(mc-5.0)  *(mc-5.0)/  (2.0*0.5*0.5))

print ydata.sum()/len(data)
print ymc.sum()/len(mc)





