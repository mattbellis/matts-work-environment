import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

import minuit

################################################################################
# Convert dictionary to kwd arguments
################################################################################
def minuit_output(m):
    parameters = m.parameters

    output = "%-2s  %-16s %14s %14s\n" % ("#","PARAMETER NAME","VALUE","ERROR")
    for n,p in zip(xrange(len(parameters)),parameters):
        output += "%-2d  %-16s %14.6e %14.6e\n" % (n,p,m.values[p],m.errors[p])

    return output

################################################################################
# Convert dictionary to kwd arguments
################################################################################
def dict2kwd(d):

    keys,vals = d.keys(),d.values()

    params_names = ()

    kwd = {}
    for k,v in d.iteritems():
        print k,v
        params_names += (k,)
        kwd[k] = v['start_val']
        if 'fix' in v and v['fix']==True:
            new_key = "fix_%s" % (k)
            kwd[new_key] = True
        if 'limits' in v:
            new_key = "limit_%s" % (k)
            kwd[new_key] = v['limits']

    ''' 
    if 'num_bkg' in keys:
        print "YES!",d['num_bkg']
    '''

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
        print "varnames"
        print varnames

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

        print "Finished with __init__"

    def __call__(self,*arg):
        #print "arg: "
        #print arg
        #print self.func_code.co_varnames
        #flag = arg[0]
        data0 = self.data[0]
        mc = self.data[1]

        val = emlf_minuit(data0,mc,arg,self.func_code.co_varnames)

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
def fitfunc(data,p,parnames):

    pn = parnames

    flag = p[pn.index('flag')]
    #mean = p[0]
    #sigma = p[1]
    #function = stats.norm(loc=mean,scale=sigma)
    #ret = function.pdf(x)

    ytot = np.zeros(len(data))

    #'''
    if flag==0:

        ytot = np.zeros(len(data))
        x = data
        
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
    #'''
    elif flag==1:

        x = data[0]
        y = data[1]

        ytot = np.zeros(len(x))
        
        mean = p[pn.index('mean')]
        sigma = p[pn.index('sigma')]
        sig_y_slope = p[pn.index('exp_sig_y')]
        bkg_x_slope = p[pn.index('exp_bkg_x')]
        num_sig = p[pn.index('num_sig')]
        num_bkg = p[pn.index('num_bkg')]

        '''
        output = ""
        for name in pn:
            output += "%-15s %12.4f\n" % (name,p[pn.index(name)])
        print output
        '''

        ########################################################################
        # Signal PDF
        ########################################################################
        gauss = stats.norm(loc=mean,scale=sigma)
        sig_exp = stats.expon(loc=0.0,scale=sig_y_slope)

        # Normalize
        xnorm = np.linspace(2.0,8.0,1000)
        ynorm = gauss.pdf(xnorm) 
        normalization = integrate.simps(ynorm,x=xnorm)
        #print "normalization: ",normalization

        xnorm = np.linspace(0.0,400.0,1000)
        ynorm = sig_exp.pdf(xnorm) 
        normalization *= integrate.simps(ynorm,x=xnorm)

        #print "normalization: ",normalization

        y = num_sig*gauss.pdf(x)*sig_exp.pdf(y)
        y /= normalization
        ytot += y 

        ########################################################################
        # Background PDF
        ########################################################################
        flat_term = 1.0
        bkg_exp = stats.expon(loc=0.0,scale=bkg_x_slope)

        # Normalize
        xnorm = np.linspace(2.0,8.0,1000)
        ynorm = bkg_exp.pdf(xnorm)
        normalization = integrate.simps(ynorm,x=xnorm)

        y = num_bkg*(1.0/400.0)*bkg_exp.pdf(x)
        #y = num_flat/6.0
        #y = num_flat*flat_term
        y /= normalization
        ytot += y 

        #print ytot

    #'''

    return ytot
################################################################################


################################################################################
# Extended maximum likelihood function for minuit
################################################################################
def emlf_minuit(data,mc,p,varnames):

    v = varnames

    #print p
    n = 0
    for name in varnames:
        if 'num_' in name:
            n += p[varnames.index(name)]

    norm_func = (fitfunc(mc,p,v)).sum()/len(mc)

    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0

    #print "ndata: ",len(data[0])
    #print "nmc  : ",len(mc[0])
    #print pois(n,len(data))
    #ret = (-np.log(fitfunc(data,p,v) / norm_func).sum()) # - pois(n,len(data))

    #ret = (-np.log(fitfunc(data,p,v))).sum() + len(data)*np.log(norm_func) - pois(n,len(data))
    ret = (-np.log(fitfunc(data,p,v))).sum() + len(data)*(norm_func) #- pois(n,len(data))
    #print ret

    return ret



