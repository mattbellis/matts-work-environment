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

bin_width = 0.017
ranges = [0.0,20.0]

################################################################################
# X-ray data fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    pdf = None
    num_wimps = 0

    x = data[0]

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

    npeaks = 14

    for i in xrange(npeaks):
        name = "ks_mean%d" % (i)
        means.append(p[pn.index(name)])
        name = "ks_sigma%d" % (i)
        sigmas.append(p[pn.index(name)])
        name = "ks_ncalc%d" % (i)
        #numks.append(p[pn.index(name)]/num_tot) # Normalized this # to number of events.
        numks.append(p[pn.index(name)]) # Normalized this # to number of events.

    for n,m,s in zip(numks,means,sigmas): 
        #pdf  = pdfs.gauss(x,m,s,xlo,xhi)
        pdf  = pdfs.lorentzian(x,m,s,xlo,xhi)
        #print "here"
        #print pdf[20:30]
        pdf *= n
        #print pdf[20:30]
        tot_pdf += pdf

    #print tot_pdf[20:30]
    #num_flat /= num_tot

    ############################################################################
    # Flat term
    ############################################################################
    pdf  = pdfs.exp(x,e_exp0,xlo,xhi)
    pdf *= num_flat
    #print "EXP"
    #print pdf
    tot_pdf += pdf

    # Is this right?
    tot_pdf *= bin_width

    #exit()

    return tot_pdf


################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def chisq_minuit(data,p,parnames,params_dict):

    x = data[0]
    y = data[1]
    yerr = data[2]

    #print x,y

    flag = p[parnames.index('flag')]

    ret =  (((fitfunc(data,p,parnames,params_dict)-y)**2)/(yerr**2)).sum()

    return ret

################################################################################




################################################################################
# Importing the data
################################################################################
#
# Full path to the directory 
infile0 = open(sys.argv[1])

xpts = np.array([])
ypts = np.array([])
xerr = np.array([])
yerr = np.array([])


for line in infile0:
    vals = line.split()
    x = float(vals[0])
    y = float(vals[1])
    xpts = np.append(xpts,x)
    ypts = np.append(ypts,y)

# Make some cuts
index0 = xpts>ranges[0]
index1 = xpts<ranges[1]
index2 = ypts>0
index = index0*index1*index2
ypts = ypts[index]
xpts = xpts[index]

yerr = np.sqrt(ypts)
xerr = np.zeros(len(yerr))

data = [xpts,ypts,yerr]


############################################################################
# Plot the data
############################################################################
fig0 = plt.figure(figsize=(12,4),dpi=100)
ax0 = fig0.add_subplot(1,1,1)

ax0.set_xlabel("Energy (keV)",fontsize=12)
ax0.set_ylabel("Counts",fontsize=12)

plt.errorbar(xpts,ypts,yerr=yerr,fmt='ko',markersize=2)
#plt.ylim(1)
#plt.yscale('log')

############################################################################

means = [7.5, 7.9, 9.0, 9.27, 10.5, 10.88, 12.57, 14.02, 14.94, 15.73, 16.7, 17.65, 17.98, 19.60]

sigmas = [0.1, 0.1, 0.1, 0.1,   0.1, 0.1, 0.1, 0.1,   0.1, 0.1, 0.1, 0.1,   0.1, 0.1]

num_decays_in_dataset = [1000,1000,1000,1000,  1000,1000,1000,1000, 1000,1000,1000,1000, 1000,1000]

############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}
#params_dict['flag'] = {'fix':True,'start_val':args.fit}
params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_e'] = {'fix':True,'start_val':0,'limits':(ranges[0],ranges[1])}

for i,val in enumerate(means):
    name = "ks_mean%d" % (i)
    params_dict[name] = {'fix':True,'start_val':val,'limits':(ranges[0],ranges[1]),'error':1}
for i,val in enumerate(sigmas):
    name = "ks_sigma%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(0.00,1.00),'error':0.1}
for i,val in enumerate(num_decays_in_dataset):
    name = "ks_ncalc%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(1.0,60000000.0),'error':1}

params_dict['num_flat'] = {'fix':False,'start_val':20000.0,'limits':(0.0,50000000.0),'error':1}
params_dict['e_exp0'] = {'fix':False,'start_val':0.5,'limits':(0.0,10.0),'error':0.1}

#plt.show()

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([data],params_dict,chisq_minuit)

m = minuit.Minuit(f,errordef=1.0,**kwd)

# For chi-square fit.
m.set_errordef(1.0)

# Up the tolerance.
m.tol = 1.0

m.migrad()

values = m.values







################################################################################
############################################################################
# Flat
############################################################################
expts = np.linspace(ranges[0],ranges[1],1000)
eytot = np.zeros(1000)


# Energy projections
#ypts = np.ones(len(expts))
ypts = np.exp(-values['e_exp0']*expts)
y,plot = plotutils.plot_pdf(expts,ypts,bin_width=bin_width,scale=values['num_flat'],fmt='m-',axes=ax0)
eytot += y

# K-shell peaks
for i,meanc in enumerate(means):
    name = "ks_mean%d" % (i)
    m = values[name]
    name = "ks_sigma%d" % (i)
    s = values[name]
    name = "ks_ncalc%d" % (i)
    n = values[name]

    #gauss = stats.norm(loc=m,scale=s)
    #eypts = gauss.pdf(expts)

    lorentzian = stats.norm(loc=m,scale=s)
    eypts = lorentzian.pdf(expts)

    # Energy distributions
    y,plot = plotutils.plot_pdf(expts,eypts,bin_width=bin_width,scale=n,fmt='r-',axes=ax0)
    eytot += y
    #lshell_totx += y


ax0.plot(expts,eytot,'b',linewidth=2)










plt.show()


