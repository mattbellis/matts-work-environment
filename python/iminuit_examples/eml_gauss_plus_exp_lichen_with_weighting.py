import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from scipy.stats import norm
from scipy.stats import expon
from math import factorial, log

import lichen.lichen as lch
import lichen.pdfs as pdfs

from iminuit import Minuit 

import lichen.iminuit_fitting_utilities as fitutils

####################################################
# Signal
# Gaussian (normal) function 
####################################################
def mygauss(x,mu,sigma):
    ret = pdfs.gauss(x,mu,sigma,0,5)
    return ret

###################################################
# Background
# Exponential
###################################################
def myexp(x,mylambda):
    ret = pdfs.exp(x,mylambda,0,5)
    return ret

##################################################
# PDF
# Gaussian and background
###################################################
def pdf(data,mu,sigma,mylambda,nsig,nbkg):
    
    # data[0] is the data points
    # data[1] is the weights
    
    ntot = float(nsig+nbkg)

    signal = data[1][0]*(nsig/ntot)*mygauss(data[0],mu,sigma)
    bkgrnd = data[1][1]*(nbkg/ntot)*myexp(data[0],mylambda)

    ret = (signal+bkgrnd)

    return ret

################################################################################
# Negative log likelihood  
################################################################################
def negative_log_likelihood(data,p,parnames,params_dict):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.

    mu = p[parnames.index('mu')]
    sigma = p[parnames.index('sigma')]
    mylambda = p[parnames.index('mylambda')]
    nsig = p[parnames.index('nsig')]
    nbkg = p[parnames.index('nbkg')]

    #print data
    ntot = len(data[0])
    pois = pdfs.pois(nsig+nbkg,ntot)

    likelihood_func =  (-np.log(pdf(data,mu,sigma,mylambda,nsig,nbkg))).sum()
    ret = likelihood_func - pois

    return ret


################################################################################
# Generate fake data points and plot them 
################################################################################
mu = 1.5
sigma = 0.1
x = np.random.normal(mu,sigma,70)
x = x[x>0]
x = x[x<5]
nsig_true = len(x)
print "# signal: %d" % (nsig_true)

# Gen weights for the signal events
wt0 = np.random.normal(1.0,1.0,len(x))


mylambda = 1.0
k = np.random.exponential(mylambda,1000)
k = k[k>0]
k = k[k<5]
nbkg_true = len(k)
print "# bkg: %d" % (nbkg_true)

# Gen background weights
wt1 = np.random.normal(4.0,1.0,len(k))

plt.figure()
#lch.hist_err(x,bins=50,range=(0,4.0),color='red',ecolor='red')
#lch.hist_err(k,bins=50,range=(0,4.0),color='blue',ecolor='blue')

data = np.append(x,k)
wts = np.append(wt0,wt1)

valsig = pdfs.gauss(wts,1.0,1.0,-5.0,15.0)
valbkg = pdfs.gauss(wts,4.0,1.0,-5.0,15.0)

#weights = [valsig/(valsig+valbkg),valbkg/(valsig+valbkg)]
weights = [valsig,valbkg]

print wt0[0:10]
print valsig[0:10]
print valbkg[0:10]
print weights[0][0:10]
print weights[1][0:10]

plt.hist(wt0,bins=20)
plt.hist(wt1,bins=20,alpha=0.3)
#plt.show()

#exit()

lch.hist_err(data,bins=50,range=(0,4.0),markersize=2)

print "min/max %f %f" % (min(data),max(data))

################################################################################
# Now fit the data.
################################################################################

params_dict = {}
params_dict['mu'] = {'fix':True,'start_val':1.5,'limits':(0,3.0),'error':0.1}
params_dict['sigma'] = {'fix':True,'start_val':0.1,'limits':(0.01,3.0),'error':0.1}
params_dict['mylambda'] = {'fix':True,'start_val':1.0,'limits':(0,10.0),'error':0.1}
params_dict['nsig'] = {'fix':False,'start_val':100,'limits':(0,1.1*len(data)),'error':0.1}
params_dict['nbkg'] = {'fix':False,'start_val':0.5*len(data),'limits':(0,1.1*len(data)),'error':0.1}

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([[data,weights]],params_dict,negative_log_likelihood)

m = Minuit(f,**kwd)

'''
m = Minuit(negative_log_likelihood,mu=1.0,limit_mu=(0,3.0), \
                                   sigma=1.0,limit_sigma=(0,3.0), \
                                   mylambda=1.0,limit_mylambda=(0,3.0), \
                                   fraction=0.5,limit_fraction=(0,3.14) \
                                   )
'''

m.migrad()
m.hesse()
#m.minos()

print 'fval', m.fval

print m.get_fmin()

values = m.values
print values

print "weights: "
print sum(weights[0]*weights[0]),sum(weights[1]*weights[1])
print sum(weights[0]),sum(weights[1])
print sum(weights[0])/sum(weights[0]*weights[0]),sum(weights[1])/sum(weights[1]*weights[1])

print nsig_true,nbkg_true

#frac_sig = np.cos(values['fraction'])**2
#print frac_sig
#print m.covariance
#print m.print_matrix()

#plt.show()
