import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

import minuit

import pdfs 

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

        ################
        params_names = ()
        #limits = {}
        kwd = {}
        for k,v in params.iteritems():
            params_names += (k,)
            '''
            if 'var_' in k and 'limits' in v:
                new_key = "%s_limits" % (k)
                limits[new_key] = v['limits']
            '''
        ################

        self.params = params_names
        self.params_dict = params
        #self.limits = 1.0
        #varnames = ['%s'%i for i in params]
        varnames = params_names
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

        val = emlf_minuit(data0,mc,arg,self.func_code.co_varnames,self.params_dict)

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
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]
    #mean = p[0]
    #sigma = p[1]
    #function = stats.norm(loc=mean,scale=sigma)
    #ret = function.pdf(x)


    #'''
    if flag==0:

        ytot = np.zeros(len(data[0]))
        x = data[0]
        
        mean = p[pn.index('mean')]
        sigma = p[pn.index('sigma')]
        num_gauss = p[pn.index('num_gauss')]
        num_flat = p[pn.index('num_flat')]

        # Surf exponential
        gauss = stats.norm(loc=mean,scale=sigma)

        xnorm = np.linspace(2.0,8.0,1000)
        ynorm = gauss.pdf(xnorm) 
        normalization = integrate.simps(ynorm,x=xnorm)
        #print "normalization: ",normalization

        y = num_gauss*gauss.pdf(x)
        #y = num_gauss*np.exp(-((x - mean)**2)/(2.0*sigma*sigma))
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

    elif flag==1:

        x = data[0]
        y = data[1]

        xlo = params_dict['var_x']['limits'][0]
        xhi = params_dict['var_x']['limits'][1]

        ylo = params_dict['var_y']['limits'][0]
        yhi = params_dict['var_y']['limits'][1]

        #print "var_x_limits: ",params_dict['var_x']['limits']
        #print "var_y_limits: ",params_dict['var_y']['limits']

        tot_pdf = np.zeros(len(x))
        
        mean = p[pn.index('mean')]
        sigma = p[pn.index('sigma')]
        sig_y_slope = p[pn.index('exp_sig_y')]
        bkg_x_slope = p[pn.index('exp_bkg_x')]
        num_sig = p[pn.index('num_sig')]
        num_bkg = p[pn.index('num_bkg')]

        ########################################################################
        # Signal PDF
        ########################################################################
        pdf  = pdfs.gauss(x,mean,sigma,xlo,xhi)
        pdf *= pdfs.exp(y,sig_y_slope,ylo,yhi)
        pdf *= num_sig

        #y /= normalization
        tot_pdf += pdf

        ########################################################################
        # Background PDF
        ########################################################################
        pdf = pdfs.poly(y,[],ylo,yhi)
        pdf *= pdfs.exp(x,bkg_x_slope,xlo,xhi)
        pdf *= num_bkg

        tot_pdf += pdf 


    return tot_pdf
################################################################################


################################################################################
# Extended maximum likelihood function for minuit
################################################################################
def emlf_minuit(data,mc,p,parnames,params_dict):

    #v = parnames

    ndata = len(data[0])
    nmc   = len(mc[0])

    #print ndata,nmc

    #print p
    #'''
    n = 0
    for name in parnames:
        if 'num_' in name:
            n += p[parnames.index(name)]
    #'''
    #print n,ndata

    norm_func = (fitfunc(mc,p,parnames,params_dict)).sum()/nmc

    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0

    #print "ndata: ",len(data[0])
    #print "nmc  : ",len(mc[0])
    #print pois(n,len(data))
    #ret = (-np.log(fitfunc(data,p,v) / norm_func).sum()) # - pois(n,len(data))

    #ret = (-np.log(fitfunc(data,p,v))).sum() + len(data)*np.log(norm_func) - pois(n,len(data))
    #print "extended term: ", ((n-ndata)*(n-ndata))/(2*ndata)
    ret = (-np.log(fitfunc(data,p,parnames,params_dict))).sum() + ndata*(norm_func) # + ((n-ndata)*(n-ndata))/(2*ndata)
    #ret = (-np.log(fitfunc(data,p,v))).sum() + n*(norm_func) 
    #print ret

    return ret



