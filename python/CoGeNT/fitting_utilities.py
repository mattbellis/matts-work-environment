import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

import time

#import minuit
import iminuit as minuit
from cogent_pdfs import *

import lichen.pdfs as pdfs

import chris_kelso_code as dmm
#import chris_kelso_code_cython as dmm

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
    # mu = # of data returned by the fit
    # k  = # of data events
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

    flag = p[parnames.index('flag')]

    # Grab all the fitting paramters.
    ranges,subranges,nbins = parameters.fitting_parameters(flag)

    wimp_model = None

    num_tot = 0
    for name in parnames:
        if flag==0 or flag==1:
            if 'num_' in name or 'ncalc' in name:
                num_tot += p[parnames.index(name)]
        elif flag==2 or flag==3 or flag==4:
            if 'num_flat' in name or 'num_exp1' in name or 'ncalc' in name:
                num_tot += p[parnames.index(name)]
    # MC
    if flag==5:
        num_tot += p[parnames.index('num_exp0')]
        num_tot += p[parnames.index('num_flat')]

    # MC
    if flag==6:
        num_tot += p[parnames.index('num_flat')]

    if flag==2 or flag==3 or flag==4 or flag==6:
        mDM = p[parnames.index('mDM')]
        sigma_n = p[parnames.index('sigma_n')]
        #loE = dmm.quench_keVee_to_keVr(0.5)
        #hiE = dmm.quench_keVee_to_keVr(3.2)
        loE = 0.5
        hiE = 3.5
        if flag==2:
            wimp_model = 'shm'
        elif flag==3:
            wimp_model = 'debris'
        elif flag==4:
            wimp_model = 'stream'
        elif flag==6:
            wimp_model = 'shm'

        #subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
        #subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]
        '''
        subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
        if flag==5 or flag==6:
            subranges = [[],[[1,917]]]
        '''

        max_val = 0.86786
        threshold = 0.345
        sigmoid_sigma = 0.241

        #eff_scaling = 1.0
        eff_scaling = 0.9 # 3yr data

        efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)/eff_scaling

        #num_wimps = 0
        '''
        for sr in subranges[1]:
            start = time.time()
            num_wimps += integrate.dblquad(wimp,loE,hiE,lambda x: sr[0],lambda x:sr[1],args=(AGe,mDM,sigma_n,efficiency,wimp_model),epsabs=dblqtol)[0]*(0.333)
            duration = time.time() - start
            print "duration: ",1000*duration

        '''
        #print "num_wimps: ",num_wimps
        #num_tot += num_wimps

    tot_pdf,num_wimps = fitfunc(data,p,parnames,params_dict)
    print "num_wimps: ",num_wimps
    if flag==2 or flag==3 or flag==4 or flag==6:
        num_tot += num_wimps

    print "pois:         %12.3f %12.3f" % (num_tot,ndata)
    #likelihood_func = (-np.log(fitfunc(data,p,parnames,params_dict))).sum()
    print tot_pdf
    for d0,d1,t in zip(data[0],data[1],tot_pdf):
        if t<0:
            print d0,d1,t,np.log(t)
    likelihood_func = (-np.log(tot_pdf)).sum()
    print "vals         : %12.3f %12.3f %12.3f" % (likelihood_func,pois(num_tot,ndata),likelihood_func-pois(num_tot,ndata))
    ret = likelihood_func - pois(num_tot,ndata)
    #print "vals         : %12.3f %12.3f %12.3f" % (likelihood_func,num_tot,likelihood_func-num_tot)
    #ret = likelihood_func - num_tot

    return ret


################################################################################
# Do contours
################################################################################
def contours(m,par0,par1,sigma=1.0,npts=5):

    print "Starting contours..."
    #print m.values
    #contour_points = m.contour(par0,par1,sigma,npts)
    #x_bins,y_bins,values = m.contour(par0,par1,bound=sigma,bins=npts,subtract_min=True)
    #x_bins,y_bins,values = m.mncontour(par0,par1,sigma=sigma,numpoints=npts)
    x,y,z,zz = m.draw_mncontour(par0,par1)
    #print contour_points
    '''
    cx = np.array([])
    cy = np.array([])
    if contour_points!=None and len(contour_points)>1:
        for p in contour_points:
            cx = np.append(cx,p[0])
            cy = np.append(cy,p[1])
        cx = np.append(cx,contour_points[0][0])
        cy = np.append(cy,contour_points[0][1])
    '''

    #return cx,cy
    #return x_bins,y_bins,values
    return x,y,z

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


