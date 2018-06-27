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

np.random.seed(100)

####################################################
# Signal
# Gaussian (normal) function 
####################################################
def mygauss(x,mu,sigma):
    ret = pdfs.gauss(x,mu,sigma,0,200)
    return ret

##################################################
# PDF
# Gaussian and background
###################################################
def pdf(data,mu,sigma,nsig):
    # p is an array of the parameters
    # x is the data points
    
    #signal = mygauss(data,mu,sigma)
    signal = (1.0/sigma)*np.exp(-((data - mu)**2)/(2.0*sigma*sigma))

    ret = signal

    return ret

################################################################################
# Negative log likelihood  
################################################################################
def negative_log_likelihood(data,p,parnames,params_dict):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.

    mu = p[parnames.index('mu')]
    sigma = p[parnames.index('sigma')]
    nsig = p[parnames.index('nsig')]

    ntot = len(data)
    pois = pdfs.pois(nsig,ntot)

    likelihood_func =  (-np.log(pdf(data,mu,sigma,nsig))).sum()
    ret = likelihood_func - pois

    return ret


################################################################################
# Generate fake data points and plot them 
################################################################################
mu = 100
sigma = 15
data = np.random.normal(mu,sigma,200)
#x = x[x>0]
#x = x[x<5]
print("# signal: %d" % (len(data)))

plt.figure()
#lch.hist_err(x,bins=50,range=(0,4.0),color='red',ecolor='red')
#lch.hist_err(k,bins=50,range=(0,4.0),color='blue',ecolor='blue')

#lch.hist_err(data,bins=50,range=(50,150.0),markersize=2)
plt.hist(data, bins=50,range=(50,150));


print("min/max %f %f" % (min(data),max(data)))

################################################################################
# Now fit the data.
################################################################################

params_dict = {}
params_dict['mu'] = {'fix':False,'start_val':120,'limits':(0,300.0),'error':0.01}
params_dict['sigma'] = {'fix':False,'start_val':18.0,'limits':(0.01,30.0),'error':0.01}
params_dict['nsig'] = {'fix':False,'start_val':200,'limits':(0,1.1*len(data)),'error':0.01}


params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([data],params_dict,negative_log_likelihood)

m = Minuit(f,**kwd)
m.set_errordef(0.5)

'''
m = Minuit(negative_log_likelihood,mu=1.0,limit_mu=(0,3.0), \
                                   sigma=1.0,limit_sigma=(0,3.0), \
                                   mylambda=1.0,limit_mylambda=(0,3.0), \
                                   fraction=0.5,limit_fraction=(0,3.14) \
                                   )
'''

print("Calculating uncertainties...")
m.migrad()
m.minos()
#m.hesse()

print('fval', m.fval)

print(m.get_fmin())

values = m.values
print(values)

errors = m.errors
print(errors)

print()
print('covariance', m.covariance)
print()
print('matrix()', m.matrix()) #covariance
print()
print('matrix(correlation=True)', m.matrix(correlation=True)) #correlation
print()
m.print_matrix() #correlation
print()
print("Here")
m.print_param()
print()
m.print_matrix()

#frac_sig = np.cos(values['fraction'])**2
#print frac_sig
#print m.covariance
#print m.print_matrix()

plt.show()





