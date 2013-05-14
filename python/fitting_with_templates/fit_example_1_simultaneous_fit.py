import numpy as np
import matplotlib.pylab as plt
import lichen.lichen as lch
import lichen.pdfs as pdfs
import lichen.iminuit_fitting_utilities as fitutils
import lichen.plotting_utilities as plotutils

import scipy.stats as stats

from datetime import datetime,timedelta

import iminuit as minuit

################################################################################
# Fit function.
################################################################################
def fitfunc(nums,templates):

    tot_pdf = np.zeros(len(templates[0]))
    for n,t in zip(nums,templates):
        tot_pdf += n*t

    return tot_pdf







################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data,p,parnames,params_dict):

    template00 = data[1][1]
    template01 = data[2][1]
    template10 = data[3][1]
    template11 = data[4][1]

    num00 = p[parnames.index('num00')]
    num01 = p[parnames.index('num01')]
    num10 = p[parnames.index('num10')]
    num11 = p[parnames.index('num11')]

    # Add up the data ``y" for the total number of events.
    #ndata0 = sum(data[0][0][1])
    #ndata1 = sum(data[0][1][1])

    flag = p[parnames.index('flag')]

    num_tot = 0.0
    for name in parnames:
        if 'num' in name:
            num_tot += p[parnames.index(name)]

    likelihood_func = 0.0

    tot_pdf0 = fitfunc([num00,num10],[template00,template10])
    tot_pdf1 = fitfunc([num01,num11],[template01,template11])

    y0 = data[0][0][1]
    y1 = data[0][1][1]
    
    likelihood_func  = (-y0*np.log(tot_pdf0)).sum()
    likelihood_func += (-y1*np.log(tot_pdf1)).sum()

    print "num_tot: ",num_tot
    #ret = likelihood_func - fitutils.pois(num_tot,ndata)
    #num_tot = num00 + num10
    ret = likelihood_func + num_tot

    return ret

################################################################################



################################################################################
# Set up a figure for future plotting.
################################################################################

fig0 = plt.figure(figsize=(8,8),dpi=100)
ax00 = fig0.add_subplot(2,2,1)
ax01 = fig0.add_subplot(2,2,2)
ax02 = fig0.add_subplot(2,2,3)
ax03 = fig0.add_subplot(2,2,4)

################################################################################
# Generate the fitting templates
################################################################################

nbins = 100
ranges = [0.0, 5.0]
ngen = 100000

# Gen sig
mu0 = 2.0
sigma0 = 0.2
data = np.random.normal(mu0,sigma0,ngen)
h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data,bins=nbins,range=ranges,axes=ax00)
norm = float(sum(ypts))
template00 = [xpts.copy(),ypts.copy()/norm]

mu1 = 2.0
sigma1 = 0.1
data = np.random.normal(mu1,sigma1,ngen)
h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data,bins=nbins,range=ranges,axes=ax01)
norm = float(sum(ypts))
template01 = [xpts.copy(),ypts.copy()/norm]

# Gen bkg
yexp = stats.expon(loc=0.0,scale=6.0)
data = yexp.rvs(ngen)
data = data[data<ranges[1]]
h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data,bins=nbins,range=ranges,axes=ax02)
norm = float(sum(ypts))
template10 = [xpts.copy(),ypts.copy()/norm]

yexp = stats.expon(loc=0.0,scale=3.0)
data = yexp.rvs(ngen)
data = data[data<ranges[1]]
h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data,bins=nbins,range=ranges,axes=ax03)
norm = float(sum(ypts))
template11 = [xpts.copy(),ypts.copy()/norm]

#print template00
#print template01
#print template10
#print template11

ax00.set_xlim(ranges[0],ranges[1])
ax01.set_xlim(ranges[0],ranges[1])
ax02.set_xlim(ranges[0],ranges[1])
ax03.set_xlim(ranges[0],ranges[1])

################################################################################
# Generate the data.
################################################################################

data0 = np.random.normal(mu0,sigma0,200)
nsig0 = len(data0)
data1 = np.random.normal(mu1,sigma1,100)
nsig1 = len(data1)

yexp = stats.expon(loc=0.0,scale=6.0)
bkg0 = yexp.rvs(400)
bkg0 = bkg0[bkg0<ranges[1]]
data0 = np.append(data0,bkg0)
nbkg0 = len(bkg0)

yexp = stats.expon(loc=0.0,scale=3.0)
bkg1 = yexp.rvs(300)
bkg1 = bkg1[bkg1<ranges[1]]
data1 = np.append(data1,bkg1)
nbkg1 = len(bkg1)

fig1 = plt.figure(figsize=(8,8),dpi=100)
ax10 = fig1.add_subplot(2,2,1)
ax11 = fig1.add_subplot(2,2,2)
ax12 = fig1.add_subplot(2,2,3)
ax13 = fig1.add_subplot(2,2,4)

h,xpts0,ypts0,xpts_err,ypts_err = lch.hist_err(data0,bins=nbins,range=ranges,axes=ax10)
h,xpts1,ypts1,xpts_err,ypts_err = lch.hist_err(data1,bins=nbins,range=ranges,axes=ax11)
ax10.set_xlim(ranges[0],ranges[1])
ax11.set_xlim(ranges[0],ranges[1])

data = [[xpts0.copy(),ypts0.copy()],[xpts1.copy(),ypts1.copy()]]

ndata0 = sum(ypts0)
ndata1 = sum(ypts1)

############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}
params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_x'] = {'fix':True,'start_val':0,'limits':(ranges[0],ranges[1])}
params_dict['num00'] = {'fix':False,'start_val':100,'limits':(0,ndata0)}
params_dict['num01'] = {'fix':False,'start_val':200,'limits':(0,ndata1)}
params_dict['num10'] = {'fix':False,'start_val':100,'limits':(0,ndata0)}
params_dict['num11'] = {'fix':False,'start_val':200,'limits':(0,ndata1)}

params_names,kwd = fitutils.dict2kwd(params_dict)

data_and_pdfs = [data,template00,template01,template10,template11]

f = fitutils.Minuit_FCN([data_and_pdfs],params_dict,emlf_normalized_minuit)

m = minuit.Minuit(f,**kwd)

# For maximum likelihood method.
m.errordef = 0.5

# Up the tolerance.
#m.tol = 1.0

m.migrad()
m.hesse()

values = m.values

print "nsig0: ",nsig0
print "nbkg0: ",nbkg0
print "nsig1: ",nsig1
print "nbkg1: ",nbkg1


'''
ax11.set_xlim(ranges[0],ranges[1])
ax11.plot(template0[0],values['num0']*template0[1],'g-',linewidth=3)
ax11.plot(template1[0],values['num1']*template1[1],'r-',linewidth=3)
ax11.plot(template1[0],values['num1']*template1[1]+values['num0']*template0[1],'b-',linewidth=3)

ax10.set_xlim(ranges[0],ranges[1])
#ax10.plot(template0[0],values['num0']*template0[1],'r',ls='steps',fill=True)
binwidth = template0[0][1]-template0[0][0]
ax10.bar(template0[0]-binwidth/2.0,values['num0']*template0[1]+values['num1']*template1[1],color='b',width=binwidth,edgecolor='b')
ax10.bar(template0[0]-binwidth/2.0,values['num1']*template1[1],color='r',width=binwidth,edgecolor='r')
'''

#plt.show()


