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

################################################################################
# WIMP signal
################################################################################
################################################################################
def wimp(y,x,AGe,mDM):
    #tc_SHM = dmm.tc(np.zeros(3))
    #print tc_SHM
    #print tc_SHM+y
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
# Cut events from an arbitrary dataset that fall outside a set of ranges.
################################################################################
def cut_events_outside_range(data,ranges):

    index = np.ones(len(data[0]),dtype=np.int)
    for i,r in enumerate(ranges):
        index *= ((data[i]>r[0])*(data[i]<r[1]))

    '''
    for x,y in zip(data[0][index!=True],data[1][index!=True]):
        print x,y
    '''

    for i in xrange(len(data)):
        print data[i][index!=True]
        data[i] = data[i][index==True]

    return data

################################################################################
# Cut events from an arbitrary dataset that fall outside a set of sub-ranges.
################################################################################
def cut_events_outside_subrange(data,subrange,data_index=0):

    index = np.zeros(len(data[data_index]),dtype=np.int)
    for r in subrange:
        print r[0],r[1]
        index += ((data[data_index]>r[0])*(data[data_index]<r[1]))
        print data[1][data[1]>107.0]

    print index[index!=1]
    for i in xrange(len(data)):
        print data[i][index!=True]
        data[i] = data[i][index==True]

    return data


################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    tot_pdf = np.zeros(len(data[0]))

    if flag==0:

        x = data[0]
        y = data[1]

        xlo = params_dict['var_e']['limits'][0]
        xhi = params_dict['var_e']['limits'][1]
        ylo = params_dict['var_t']['limits'][0]
        yhi = params_dict['var_t']['limits'][1]

        tot_pdf = np.zeros(len(x))
        
        e_exp0 = p[pn.index('e_exp0')]
        num_exp0 = p[pn.index('num_exp0')]
        num_flat = p[pn.index('num_flat')]
        e_exp1 = p[pn.index('e_exp1')]
        num_exp1 = p[pn.index('num_exp1')]

        #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)
        #lshells = lshell_peaks(means,sigmas,num_decays_in_dataset)

        means = []
        sigmas = []
        numls = []
        decay_constants = []

        for i in xrange(10):
            name = "ls_mean%d" % (i)
            means.append(p[pn.index(name)])
            name = "ls_sigma%d" % (i)
            sigmas.append(p[pn.index(name)])
            name = "ls_ncalc%d" % (i)
            numls.append(p[pn.index(name)])
            name = "ls_dc%d" % (i)
            decay_constants.append(p[pn.index(name)])

        for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
            pdf  = pdfs.gauss(x,m,s,xlo,xhi)
            #dc = -1.0/dc
            dc = -1.0*dc
            pdf *= pdfs.exp(y,dc,ylo,yhi)
            pdf *= n
            tot_pdf += pdf

        # Exponential in energy
        pdf  = pdfs.poly(y,[],ylo,yhi)
        pdf *= pdfs.exp(x,e_exp0,xlo,xhi)
        pdf *= num_exp0
        tot_pdf += pdf

        # Second exponential in energy
        pdf  = pdfs.poly(y,[],ylo,yhi)
        pdf *= pdfs.exp(x,e_exp1,xlo,xhi)
        pdf *= num_exp1
        tot_pdf += pdf

        # Flat term
        #print xlo,xhi,ylo,yhi
        pdf  = pdfs.poly(y,[],ylo,yhi)
        pdf *= pdfs.poly(x,[],xlo,xhi)
        pdf *= num_flat
        tot_pdf += pdf

    elif flag==1:

        max_val = 0.86786
        threshold = 0.345
        sigmoid_sigma = 0.241

        efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)
        #efficiency = lambda x: 1.0

        #subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
        subranges = [[],[[1,459]]]

        x = data[0]
        y = data[1]

        xlo = params_dict['var_e']['limits'][0]
        xhi = params_dict['var_e']['limits'][1]
        ylo = params_dict['var_t']['limits'][0]
        yhi = params_dict['var_t']['limits'][1]

        tot_pdf = np.zeros(len(x))
        
        e_exp0 = p[pn.index('e_exp0')]
        num_exp0 = p[pn.index('num_exp0')]
        num_flat = p[pn.index('num_flat')]
        e_exp1 = p[pn.index('e_exp1')]
        num_exp1 = p[pn.index('num_exp1')]

        #wmod_freq = p[pn.index('wmod_freq')]
        #wmod_phase = p[pn.index('wmod_phase')]
        #wmod_amp = p[pn.index('wmod_amp')]
        #wmod_offst = p[pn.index('wmod_offst')]

        #mDM = 7.0
        mDM = p[pn.index('mDM')]

        loE = dmm.quench_keVee_to_keVr(0.5)
        hiE = dmm.quench_keVee_to_keVr(3.2)

        # Normalize numbers.
        num_tot = 0.0
        for name in pn:
            if 'num_flat' in name or 'num_exp1' in name or 'ncalc' in name:
                num_tot += p[pn.index(name)]
                #print "building num_tot",num_tot,p[pn.index(name)]

        num_wimps = integrate.dblquad(wimp,loE,hiE,lambda x: 1.0, lambda x: 459.0,args=(AGe,mDM))[0]*(0.333)*(0.867)
        num_tot += num_wimps

        num_exp0 /= num_tot
        num_exp1 /= num_tot
        num_flat /= num_tot

        #tot_pct = num_exp0 + num_exp1 + num_flat
        #print "num_tot",num_tot
        #means,sigmas,num_decays,num_decays_in_dataset,decay_constants = lshell_data(442)
        #lshells = lshell_peaks(means,sigmas,num_decays_in_dataset)

        means = []
        sigmas = []
        numls = []
        decay_constants = []

        for i in xrange(11):
            name = "ls_mean%d" % (i)
            means.append(p[pn.index(name)])
            name = "ls_sigma%d" % (i)
            sigmas.append(p[pn.index(name)])
            name = "ls_ncalc%d" % (i)
            numls.append(p[pn.index(name)]/num_tot) # Normalized this
                                                    # to number of events.
            name = "ls_dc%d" % (i)
            decay_constants.append(p[pn.index(name)])

        for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
            pdf  = pdfs.gauss(x,m,s,xlo,xhi,efficiency=efficiency)
            dc = -1.0*dc
            pdf *= pdfs.exp(y,dc,ylo,yhi,subranges=subranges[1])
            pdf *= n
            #tot_pct += n
            tot_pdf += pdf

        ########################################################################
        # Wimp-like signal
        ########################################################################
        #pdf  = pdfs.exp(x,e_exp0,xlo,xhi,efficiency=efficiency)
        #pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
        #pdf *= pdfs.cos(y,wmod_freq,wmod_phase,wmod_amp,wmod_offst,ylo,yhi,subranges=subranges[1])
        #tc_SHM = dmm.tc(np.zeros(3))
        #gdbl_int = integrate.dblquad(wimp,loE,hiE,lambda x: 1.0, lambda x: 459.0,args=(AGe,mDM))
        gdbl_int = (1.0,1.0)
        print "gdbl_int: ",gdbl_int,mDM
        xkeVr = dmm.quench_keVee_to_keVr(x)
        pdf = dmm.dRdErSHM(xkeVr,tc_SHM+y,AGe,mDM)/gdbl_int[0]
        print "here"
        pdf *= num_exp0
        tot_pdf += pdf

        # Second exponential in energy
        pdf  = pdfs.exp(x,e_exp1,xlo,xhi,efficiency=efficiency)
        pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
        pdf *= num_exp1
        tot_pdf += pdf

        # Flat term
        #print xlo,xhi,ylo,yhi
        pdf  = pdfs.poly(x,[],xlo,xhi,efficiency=efficiency)
        pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
        pdf *= num_flat
        tot_pdf += pdf

        #print "tot_pct: ",tot_pct

    return tot_pdf
################################################################################




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
        if 'num_flat' in name or 'num_exp1' in name or 'ncalc' in name:
            n += p[parnames.index(name)]

    mDM = p[parnames.index('mDM')]

    loE = dmm.quench_keVee_to_keVr(0.5)
    hiE = dmm.quench_keVee_to_keVr(3.2)

    nwimps = integrate.dblquad(wimp,loE,hiE,lambda x: 1.0, lambda x: 459.0,args=(AGe,mDM))[0]*(0.333)*(0.867)
    n += nwimps
    print "nwimps: ",nwimps

    #print "pois: ",n,ndata
    #print "vals: ",(-np.log(fitfunc(data,p,parnames,params_dict))).sum(), pois(n,ndata)
    ret = (-np.log(fitfunc(data,p,parnames,params_dict))).sum() - pois(n,ndata)

    return ret


