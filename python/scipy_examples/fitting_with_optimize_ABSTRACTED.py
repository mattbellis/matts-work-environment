import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import fmin_bfgs,fmin_l_bfgs_b
from scipy_fitting_tools import Parameter,get_numbers,reset_parameters,pois,errfunc

import numpy as np

np.random.seed(0)


################################################################################
def signal(pars, x, frange=None):

    #print("signal: ")
    #print(pars)
    #mean,sigma = pars
    #print(pars)
    mean = pars["signal"]["mean"].value
    sigma = pars["signal"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################

################################################################################
def background(x, frange=None):

    # Flat
    height = 1.0/(frange[1] - frange[0])

    pdfvals = height*np.ones(len(x))

    return pdfvals
################################################################################

################################################################################
def pdf(pars,x,frange=None):

    nsig = pars["signal"]["number"].value
    nbkg = pars["bkg"]["number"].value

    ntot = float(nsig + nbkg)

    sig = signal(pars,x,frange=frange)
    bkg = background(x,frange=frange)

    totpdf = (nsig/ntot)*sig + (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Trying something
################################################################################
testpars = {}
testpars["signal"] = {"number":Parameter(1000,(0,2000)), "mean":Parameter(5.0,(0.1,10.0)), "sigma":Parameter(0.5,(0.1,1.0))}
testpars["bkg"] = {"number":Parameter(1000,(0,2000))}

print(testpars)

x = Parameter(10,(0,10))

print(x)
print(x.value)
print(x.limits)

nums = get_numbers(testpars)
print(nums,sum(nums))

p0 = []
parbounds = []
mapping = []
for key in testpars:
    print(testpars[key])
    for k in testpars[key].keys():
        p0.append(testpars[key][k].value)
        parbounds.append(testpars[key][k].limits)
        mapping.append((key,k))

print("p0")
print(p0)
print(mapping)
testpars['mapping'] = mapping
#exit()
################################################################################


data = stats.norm(testpars["signal"]["mean"].value,testpars["signal"]["sigma"].value).rvs(size=1000).tolist()
data += (10*np.random.random(1000)).tolist()

fix_or_float = []
'''
for p in p0:
    fix_or_float.append(None)
'''

#print(fix_or_float)
print("Starting...")
print(p0)
#p1 = fmin_bfgs(errfunc, p0, args=(data, data, fix_or_float), maxiter=100, full_output=True)#, retall=True)
p1 = fmin_l_bfgs_b(errfunc, p0, args=(data, data, fix_or_float, testpars,pdf), bounds=parbounds, approx_grad=True)#, maxiter=100 )#,factr=0.1)
print("Ending...")

print(p1)
finalvals = p1[0]
reset_parameters(testpars,finalvals)
#'''

#exit()



xpts = np.linspace(0,10,1000)

#total = pdf(pars,xpts,frange=(0,10))
#plt.figure()
#plt.plot(xpts,total)
#plt.ylim(0)

plt.figure()
binwidth=(10/100)
plt.hist(data,bins=100,range=(0,10))

ysig = testpars['signal']['number'].value*signal(testpars,xpts) * binwidth
plt.plot(xpts,ysig,linewidth=3)

ybkg = testpars['bkg']['number'].value*np.ones(len(xpts))/(10-0) * binwidth
plt.plot(xpts,ybkg,linewidth=3)

##plt.plot(xpts,ybkg + ysig,linewidth=3)
ntot = sum(get_numbers(testpars))
ytot = ntot*pdf(testpars,xpts,frange=(0,10)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
