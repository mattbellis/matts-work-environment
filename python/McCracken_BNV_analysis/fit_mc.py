import numpy as np
import matplotlib.pylab as plt
import lichen.lichen as lch
import lichen.pdfs as pdfs
import lichen.iminuit_fitting_utilities as fitutils
import lichen.plotting_utilities as plotutils

import scipy.stats as stats

from datetime import datetime,timedelta

import iminuit as minuit

import sys

nbins = [120]
ranges = [[1.05,1.16]]
bin_widths = np.ones(len(ranges))
for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
    bin_widths[i] = (r[1]-r[0])/n

################################################################################
# CoGeNT fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    pdf = None

    x_temp = data[0]
    index0 = data[0]>ranges[0][0]
    index1 = data[0]<ranges[0][1]
    x = x_temp[index0*index1]


    xlo = params_dict['var_e']['limits'][0]
    xhi = params_dict['var_e']['limits'][1]

    tot_pdf = np.zeros(len(x))

    num_flat = p[pn.index('num_flat')]
    e_exp0 = p[pn.index('e_exp0')]

    ############################################################################
    # k-shell peaks
    ############################################################################
    means = []
    sigmas = []
    numks = []
    decay_constants = []

    npeaks = 2

    num_tot = 0.0
    for name in parnames:
        if 'num_' in name or 'ncalc' in name:
            num_tot += p[parnames.index(name)]
        elif 'num_flat' in name or 'num_exp1' in name:
            num_tot += p[parnames.index(name)]

    #print "num_tot: ",num_tot

    for i in xrange(npeaks):
        name = "ks_mean%d" % (i)
        means.append(p[pn.index(name)])
        name = "ks_sigma%d" % (i)
        sigmas.append(p[pn.index(name)])
        name = "ks_ncalc%d" % (i)
        numks.append(p[pn.index(name)]/num_tot) # Normalized this
                                                # to number of events.
    #name = "ls_dc%d" % (i)
    #decay_constants.append(p[pn.index(name)])

    for n,m,s in zip(numks,means,sigmas): 
        pdf  = pdfs.gauss(x,m,s,xlo,xhi)
        #dc = -1.0*dc
        #pdf *= pdfs.exp(y,dc,ylo,yhi,subranges=subranges[1])
        pdf *= n
        tot_pdf += pdf

    num_flat /= num_tot

    ############################################################################
    # Flat term
    ############################################################################
    #pdf  = pdfs.poly(x,[],xlo,xhi)
    pdf  = pdfs.exp(x,e_exp0,xlo,xhi)
    #pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
    pdf *= num_flat
    #print "flat pdf: ",pdf[0:8]/num_flat
    #print "flat pdf: ",pdf[0:8]
    tot_pdf += pdf

    return tot_pdf



################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data,p,parnames,params_dict):

    ndata = len(data[0])

    flag = p[parnames.index('flag')]

    num_tot = 0.0
    for name in parnames:
        if 'num_' in name or 'ncalc' in name:
            num_tot += p[parnames.index(name)]
        elif 'num_flat' in name or 'num_exp1' in name:
            num_tot += p[parnames.index(name)]

    tot_pdf = fitfunc(data,p,parnames,params_dict)

    likelihood_func = (-np.log(tot_pdf)).sum()

    #print num_tot,ndata

    ret = likelihood_func - fitutils.pois(num_tot,ndata)

    return ret

################################################################################



################################################################################
# Importing the data
################################################################################
#
# Full path to the directory 
infile0 = open(sys.argv[1])

data = np.loadtxt(infile0)
index0 = data>ranges[0][0]
index1 = data<ranges[0][1]
xpts = data[index0*index1]

data = [xpts]


############################################################################
# Plot the data
############################################################################
fig0 = plt.figure(figsize=(8,4),dpi=100)
ax0 = fig0.add_subplot(1,1,1)
fig0.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.12,wspace=0.05,hspace=0.05)

ax0.set_xlabel(r"Missing mass off $K^{+}$ GeV/c$^{2}$",fontsize=18)
name = r"Entries/%0.2f MeV/c$^{2}$" % (1000*bin_widths[0])
ax0.set_ylabel(name,fontsize=18)

print data[0]
lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)

############################################################################

means = [1.115,1.115]

sigmas = [0.005,0.100]

num_decays_in_dataset = [10000,1000]

############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}

params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_e'] = {'fix':True,'start_val':0,'limits':(ranges[0][0],ranges[0][1])}

for i,val in enumerate(means):
    name = "ks_mean%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(ranges[0][0],ranges[0][1]),'error':1}
for i,val in enumerate(sigmas):
    name = "ks_sigma%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(0.00,1.00),'error':0.001}
for i,val in enumerate(num_decays_in_dataset):
    name = "ks_ncalc%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(1.0,60000000.0),'error':1}

params_dict['num_flat'] = {'fix':False,'start_val':10.0,'limits':(0.0,50000000.0),'error':1}
params_dict['e_exp0'] = {'fix':False,'start_val':0.5,'limits':(0.0,10.0),'error':0.1}

#plt.show()

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([data],params_dict,emlf_normalized_minuit)

m = minuit.Minuit(f,errordef=0.5,print_level=1,**kwd)

# Up the tolerance.
m.tol = 1.0

m.migrad()
m.hesse()

values = m.values







############################################################################
# Flat
############################################################################
expts = np.linspace(ranges[0][0],ranges[0][1],1000)
eytot = np.zeros(1000)


# Energy projections
#ypts = np.ones(len(expts))
ypts = np.exp(-values['e_exp0']*expts)
y,plot = plotutils.plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0)
eytot += y

# K-shell peaks
for i,meanc in enumerate(means):
    name = "ks_mean%d" % (i)
    m = values[name]
    name = "ks_sigma%d" % (i)
    s = values[name]
    name = "ks_ncalc%d" % (i)
    n = values[name]

    gauss = stats.norm(loc=m,scale=s)
    eypts = gauss.pdf(expts)

    # Energy distributions
    y,plot = plotutils.plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=n,fmt='r-',axes=ax0)
    eytot += y
    #lshell_totx += y


ax0.plot(expts,eytot,'b',linewidth=2)
ax0.set_xlim(ranges[0][0],ranges[0][1])


plt.show()


