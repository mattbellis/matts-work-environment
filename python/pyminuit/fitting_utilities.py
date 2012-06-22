import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from RTMinuit import *


################################################################################
# Convert dictionary to kwd arguments
################################################################################
def dict2kwd(d):

    keys,vals = d.keys(),d.values()

    params_names = d.keys()
    kwd = {}
    for k,v in d.iteritems():
        print k,v
        kwd[k] = v['start_val']
        if 'fix' in v and v['fix']==True:
            new_key = "fix_%s" % (k)
            kwd[new_key] = True
        if 'range' in v:
            new_key = "limit_%s" % (k)
            kwd[new_key] = v['range']

    if 'num_bkg' in keys:
        print "YES!",d['num_bkg']

    return params_names,kwd

################################################################################
# Sigmoid function.
################################################################################
def sigmoid(x,thresh,sigma,max_val):

    ret = max_val / (1.0 + np.exp(-(x-thresh)/(thresh*sigma)))

    return ret


################################################################################
# Helper function
################################################################################
class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

################################################################################
# Helper fitting function
################################################################################
class Minuit_FCN:
    def __init__(self,data,params):
        self.data = data
        self.params = params
        #varnames = ['%s'%i for i in params]
        varnames = params

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

    def __call__(self,*arg):
        #print "arg: "
        #print arg
        #print self.func_code.co_varnames
        flag = arg[0]
        data0 = self.data[0]
        mc = self.data[1]

        val = emlf_minuit(data0,mc,arg[1:],flag,self.func_code.co_varnames[1:])

        return val

################################################################################

################################################################################
# Poisson function
################################################################################
def pois(mu, k):
    ret = -mu + k*np.log(mu)
    return ret
################################################################################

################################################################################
def fitfunc(x,p,flag,parnames):

    pn = parnames
    #mean = p[0]
    #sigma = p[1]
    #function = stats.norm(loc=mean,scale=sigma)
    #ret = function.pdf(x)

    ytot = np.zeros(len(x))

    #'''
    if flag==0:
        
        mean = p[pn.index('mean')]
        sigma = p[pn.index('sigma')]
        num_gauss = p[pn.index('num_gauss')]
        num_flat = p[pn.index('num_flat')]

        # Surf exponential
        gauss = stats.norm(loc=mean,scale=sigma)

        xnorm = np.linspace(2.0,8.0,1000)
        ynorm = gauss.pdf(xnorm) 
        normalization = integrate.simps(ynorm,x=xnorm)
        print "normalization: ",normalization

        y = num_gauss*gauss.pdf(x)
        y /= normalization
        ytot += y 

        #'''
        # Flat
        flat_term = 1.0

        #xnorm = np.linspace(0.5,3.2,1000)
        #ynorm = 1.0*np.ones(len(xnorm))
        #normalization = integrate.simps(ynorm,x=xnorm)

        y = num_flat/6.0
        #y = num_flat*flat_term
        #y /= normalization
        ytot += y 

        #print ytot

    #'''

    return ytot
################################################################################


################################################################################
# Extended maximum likelihood function for minuit
################################################################################
def emlf_minuit(data,mc,p,flag,varnames):

    v = varnames
    norm_func = (fitfunc(mc,p,flag,v)).sum()/len(mc)

    print p
    n = 0
    for name in varnames:
        if 'num_' in name:
            n += p[varnames.index(name)]
    print "tot num: ",n

    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0

    print len(data)
    #print pois(n,len(data))
    #ret = (-np.log(fitfunc(data,p,flag,v) / norm_func).sum()) # - pois(n,len(data))

    ret = (-np.log(fitfunc(data,p,flag,v))).sum() + len(data)*np.log(norm_func) - pois(n,len(data))
    #print ret

    return ret



