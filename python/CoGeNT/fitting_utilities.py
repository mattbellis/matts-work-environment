import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

import minuit
from cogent_pdfs import *

import lichen.pdfs as pdfs

import chris_kelso_code as dmm

tc_SHM = dmm.tc(np.zeros(3))
AGe = 72.6

dblqtol = 1.0

################################################################################
# Plot WIMP signal
################################################################################
def plot_wimp_er(x,AGe,mDM,time_range=[1,365]):
    n = 0
    keVr = dmm.quench_keVee_to_keVr(x)
    for org_day in range(time_range[0],time_range[1],1):
        day = (org_day+338)%365.0 - 151
        n += dmm.dRdErSHM(keVr,tc_SHM+day,AGe,mDM)
    return n

def plot_wimp_day(org_day,AGe,mDM,e_range=[0.5,3.2]):
    n = 0
    day = (org_day+338)%365.0 - 151
    #day = org_day
    #print day
    if type(day)==np.ndarray:
        #print "here!"
        n = np.zeros(len(day))
        for i,d in enumerate(day):
            #print d
            x = np.linspace(e_range[0],e_range[1],100)
            keVr = dmm.quench_keVee_to_keVr(x)
            n[i] = (dmm.dRdErSHM(keVr,tc_SHM+d,AGe,mDM)).sum()
            #print len(tot)
    else:
        for x in np.linspace(e_range[0],e_range[1],100):
            keVr = dmm.quench_keVee_to_keVr(x)
            n += dmm.dRdErSHM(keVr,tc_SHM+day,AGe,mDM)
    return n

################################################################################
# WIMP signal
################################################################################
################################################################################
def wimp(org_day,x,AGe,mDM):
    #tc_SHM = dmm.tc(np.zeros(3))
    #print tc_SHM
    #print tc_SHM+y
    y = (org_day+338)%365.0 - 151
    dR = dmm.dRdErSHM(x,tc_SHM+y,AGe,mDM)
    return dR
################################################################################




################################################################################
# Generate some flat Monte Carlo over the range.
################################################################################
def gen_mc(nmc,ranges):
    
    mc = np.array([])
    ndim = len(ranges)

    for i in xrange(ndim):
        mc = np.append(mc,None)

    for i,r in enumerate(ranges):
        mc[i] = (r[1]-r[0])*np.random.random(nmc) + r[0]

    return mc





################################################################################
################################################################################
################################################################################
################################################################################
# Return the numbers of events
################################################################################
def return_numbers_of_events(m,acc_integral,nacc,raw_integral,nraw,num_data,params_to_use=None):

    # Total up the number of events in the dataset, returned by the fit.
    num_parameters = []
    tot_fit_events = 0.0
    for k,v in m.values.iteritems():
        if "num_" in k:
            if params_to_use==None or k in params_to_use:
                tot_fit_events += v
                num_parameters.append(k)

    tot_data_events = 0.0
    number_of_events = {}
    for name in num_parameters:
        num = m.values[name]
        err = m.errors[name]
        pct_num = num/float(tot_fit_events)
        pct_err = err/num
        number_of_events[name] = {"pct":pct_num,"pct_err":pct_err,"ndata":pct_num*num_data,"ndata_err":pct_num*num_data*pct_err}

    acc_corr_term = (float(nraw)/float(nacc))*(1.0/float(nraw))*raw_integral

    number_of_events['total'] = {"pct":1.0,"ndata":num_data,"nacc_corr":acc_corr_term*num_data}
    tot_pct_err = 0.0
    for name in num_parameters:
        nd = number_of_events[name]["ndata"]
        pct_err = number_of_events[name]["pct_err"]
        #tot_pct_err += (pct
        number_of_events[name]["nacc_corr"] = nd*acc_corr_term
        number_of_events[name]["nacc_corr_err"] = nd*acc_corr_term*pct_err

    return number_of_events

################################################################################
# Convert dictionary to kwd arguments
################################################################################
def minuit_output(m):
    parameters = m.parameters

    output = "FIXED PARAMETERS\n"
    output += "%-2s  %-16s %14s %14s\n" % ("#","PARAMETER NAME","VALUE","ERROR")
    for n,p in zip(xrange(len(parameters)),parameters):
        if m.fixed[p]==True:
            output += "%-2d  %-16s %14.6e %14.6e\n" % (n,p,m.values[p],m.errors[p])

    output += "\nFREE PARAMETERS\n"
    output += "%-2s  %-16s %14s %14s\n" % ("#","PARAMETER NAME","VALUE","ERROR")
    for n,p in zip(xrange(len(parameters)),parameters):
        if m.fixed[p]==False:
            output += "%-2d  %-16s %14.6e %14.6e\n" % (n,p,m.values[p],m.errors[p])

    return output

################################################################################
# Convert dictionary to kwd arguments
################################################################################
def dict2kwd(d,verbose=False):

    keys,vals = d.keys(),d.values()

    params_names = ()

    kwd = {}
    for k,v in d.iteritems():
        if verbose:
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
        #print "varnames"
        #print varnames

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

        print "Finished with __init__"

    def __call__(self,*arg):
        #print "arg: "
        #print arg
        #print self.func_code.co_varnames
        #flag = arg[0]
        data0 = self.data[0]
        #mc = self.data[1]

        #val = emlf_minuit(data0,mc,arg,self.func_code.co_varnames,self.params_dict)
        val = emlf_normalized_minuit(data0,arg,self.func_code.co_varnames,self.params_dict)

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
# Extended maximum likelihood function for minuit
################################################################################
def emlf_minuit(data,mc,p,parnames,params_dict):

    #v = parnames

    ndata = len(data[0])
    nmc   = len(mc[0])

    #print ndata,nmc

    #print p
    #'''
    ntot = 0
    for name in parnames:
        if 'num_' in name or 'ncalc' in name:
            ntot += p[parnames.index(name)]
    #'''
    print ntot,ndata

    norm_func = (fitfunc(mc,p,parnames,params_dict)).sum()/nmc

    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0

    #print "extended term: ", ((ntot-ndata)*(ntot-ndata))/(2*ndata)
    #ret = (-np.log(fitfunc(data,p,parnames,params_dict))).sum() + ndata*(norm_func) # + ((ntot-ndata)*(ntot-ndata))/(2*ndata)
    ret = (-np.log(fitfunc(data,p,parnames,params_dict))).sum() + ndata*(norm_func) #+ ((ntot-ndata)*(ntot-ndata))/(0.000001)

    return ret



################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data,p,parnames,params_dict):

    ndata = len(data[0])

    n = 0
    for name in parnames:
        #if 'num_flat' in name or 'num_exp1' in name or 'ncalc' in name:
        if 'num_' in name or 'ncalc' in name:
            n += p[parnames.index(name)]

    #mDM = p[parnames.index('mDM')]

    loE = dmm.quench_keVee_to_keVr(0.5)
    hiE = dmm.quench_keVee_to_keVr(3.2)

    #nwimps = integrate.dblquad(wimp,loE,hiE,lambda x: 1.0, lambda x: 459.0,args=(AGe,mDM),epsabs=dblqtol)[0]*(0.333)*(0.867)
    #n += nwimps
    #print "nwimps: ",nwimps

    #print "pois: ",n,ndata
    #print "vals: ",(-np.log(fitfunc(data,p,parnames,params_dict))).sum(), pois(n,ndata)
    ret = (-np.log(fitfunc(data,p,parnames,params_dict))).sum() - pois(n,ndata)

    return ret


################################################################################
# Do contours
################################################################################
def contours(m,par0,par1,sigma=1.0,npts=5):

    print "Starting contours..."
    #print m.values
    contour_points = m.contour(par0,par1,sigma,npts)
    #print contour_points
    cx = np.array([])
    cy = np.array([])
    if contour_points!=None and len(contour_points)>1:
        for p in contour_points:
            cx = np.append(cx,p[0])
            cy = np.append(cy,p[1])
        cx = np.append(cx,contour_points[0][0])
        cy = np.append(cy,contour_points[0][1])

    return cx,cy

################################################################################
# Do contours
################################################################################
def print_correlation_matrix(m):

    print '---------------------'
    print "\nCorrelation matrix"
    print "\nm.matrix()"
    print m.matrix(correlation=True)
    corr_matrix = m.matrix(correlation=True)
    output = ""
    for i in xrange(len(corr_matrix)):
        for j in xrange(len(corr_matrix[i])):
            output += "%9.2e " % (corr_matrix[i][j])
        output += "\n"
    print output

################################################################################
# Do contours
################################################################################
def print_covariance_matrix(m):

    print '---------------------'
    print "\nCorrelation matrix"
    print "\nm.covariance"

    print m.covariance
    cov_matrix = m.covariance
    output = ""
    for i in params_names:
        for j in params_names:
            key = (i,j)
            if key in cov_matrix:
                #output += "%11.2e " % (cov_matrix[key])
                output += "%-12s %-12s %11.4f\n" % (i,j,cov_matrix[key])
        #output += "\n"
    print output


