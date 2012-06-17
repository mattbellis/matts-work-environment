import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from RTMinuit import *
from cogent_pdfs import *


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
        varnames = ['%s'%i for i in params]

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

    def __call__(self,*arg):
        print "arg: "
        print arg
        print self.func_code.co_varnames
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

    if flag==0:
        
        exp_slope = p[pn.index('exp_slope')]
        num_exp = p[pn.index('num_exp')]
        num_flat = p[pn.index('num_flat')]

        means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)
        lshells = lshell_peaks(means,sigmas,num_decays_in_dataset)

        tot_cp = 0
        for n,cp in zip(num_decays_in_dataset,lshells):
            #y = n*cp.pdf(x)*bin_width*efficiency/HG_trigger
            y = n*cp.pdf(x)
            tot_cp += n
            ytot += y

        #p[pn.index('num_lshell')] = tot_cp

        # Surf exponential
        surf_expon = stats.expon(scale=1.0)
        y = num_exp*surf_expon.pdf(exp_slope*x)
        ytot += y 

        # Surf exponential
        flat_term = 1.0
        y = num_flat*flat_term
        ytot += y 

        #print ytot


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
    print n

    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0


    print len(data)
    print pois(n,len(data))
    ret = (-np.log(fitfunc(data,p,flag,v) / norm_func).sum()) - pois(n,len(data))
    #print ret

    return ret



