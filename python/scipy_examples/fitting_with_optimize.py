import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import fmin_bfgs,fmin_l_bfgs_b

import numpy as np

from collections import OrderedDict

np.random.seed(0)

################################################################################
def signal(pars, x, frange=None):

    print("signal: ")
    print(pars)
    mean,sigma = pars

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
def pois(mu, k):
    #mu = p[0]
    ret = -mu + k*np.log(mu)
    return ret
################################################################################

################################################################################
def total(pars,x,frange=None):
    print("frange: ")
    print(frange)

    ntot = len(x)
    functions = []

    print("total: ")
    print(pars)
    mean = pars[0]
    sigma = pars[1]
    nsig = pars[2]
    nbkg = pars[3]

    ntot = float(nsig + nbkg)

    '''
    functions.append(signal(x,[mean,sigma],frange=frange))
    functions.append(background(x,frange=frange))
    totpdf = np.zeros(ntot)
    for f in functions:
        totpdf += f
    '''
    sig = signal([mean,sigma],x,frange=frange)
    bkg = background(x,frange=frange)

    totpdf = (nsig/ntot)*sig + (nbkg/ntot)*bkg

    return totpdf
################################################################################


################################################################################
def fitfunc(p, x):
  ret = (1.0/(p[1]*np.sqrt(2*np.pi)))*np.exp(-((x - p[0])**2)/(2.0*p[1]*p[1]))
  return ret
################################################################################

####################################################
# Extended maximum likelihood method 
# This is the NLL which will be minimized
####################################################
def errfunc(pars, x, y, fix_or_float=[],params_dictionary=None):
  ret = None

  ##############################################################################
  # Update the dictionary with the new parameter values
  mapping = params_dictionary["mapping"]
  for i,m in enumerate(mapping):
      params_dictionary[m[0]][m[1]] = pars[i]

  print(params_dictionary)
  ##############################################################################

  print("------- in errfunc --------")
  print(pars)

  newpars = []
  if len(fix_or_float)==0:
    newpars = pars

  elif len(fix_or_float)==len(pars) + 1:
    pcount = 0
    for val in fix_or_float:
      if val is None:
        newpars.append(pars[pcount])
        pcount += 1
      else:
        newpars.append(val)

  # nums = get_numbers(params_dictionary)
  # ntot = sum(nums)
  nsig = pars[2]
  nbkg = pars[3]
  ntot = nsig + nbkg
        
  print("newpars: ")
  print(newpars)
  #ret = (-np.log(fitfunc(newpars, x)) .sum()) - pois(newpars, len(x))
  ret = (-np.log(total(newpars, x, frange=(0,10))) .sum()) - pois(ntot, len(x))
  print("NLL: ",ret)
  
  return ret
################################################################################

################################################################################
# Trying something
################################################################################
testpars = OrderedDict()
testpars["signal"] = OrderedDict({"number":500, "mean":1.0, "sigma":0.5})
testpars["bkg"] = OrderedDict({"number":500})

def get_numbers(params_dictionary):
    numbers = []
    for key in testpars:
        print(testpars[key])
        for k in testpars[key].keys():
            if k=="number":
                numbers.append(testpars[key][k])
    return numbers

nums = get_numbers(testpars)
print(nums,sum(nums))

p0 = []
mapping = []
for key in testpars:
    print(testpars[key])
    for k in testpars[key].keys():
        p0.append(testpars[key][k])
        mapping.append((key,k))

print(p0)
print(mapping)
testpars['mapping'] = mapping
#exit()
################################################################################

pars = {}
pars["mean"] = 5.0
pars["sigma"] = 0.5

data = stats.norm(pars["mean"],pars["sigma"]).rvs(size=1000).tolist()
data += (10*np.random.random(1000)).tolist()

#'''
p0 = [6, 0.6, 1000, len(data)-1000]
par_bounds = [(0,10), (0.1,0.9), (0,2000), (0,2000)]

fix_or_float = []
'''
for p in p0:
    fix_or_float.append(None)
'''

#print(fix_or_float)
print("Starting...")
print(p0)
#p1 = fmin_bfgs(errfunc, p0, args=(data, data, fix_or_float), maxiter=100, full_output=True)#, retall=True)
p1 = fmin_l_bfgs_b(errfunc, p0, args=(data, data, fix_or_float, testpars), bounds=par_bounds, approx_grad=True)#, maxiter=100 )#,factr=0.1)
print("Ending...")

print(p1)
finalvals = p1[0]
#'''

#exit()



xpts = np.linspace(0,10,1000)
#'''
pdf = signal([5,0.02],xpts)
#'''

#pdf = total(pars,xpts,frange=(0,10))

#plt.figure()
#plt.plot(xpts,pdf)
#plt.ylim(0)

plt.figure()
binwidth=(10/100)
plt.hist(data,bins=100,range=(0,10))

ysig = finalvals[2]*signal(finalvals[0:2],xpts) * binwidth
plt.plot(xpts,ysig,linewidth=3)

ybkg = finalvals[3]*np.ones(len(xpts))/(10-0) * binwidth
plt.plot(xpts,ybkg,linewidth=3)

#plt.plot(xpts,ybkg + ysig,linewidth=3)
ytot = (finalvals[2] + finalvals[3])*total(finalvals,xpts,frange=(0,10)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
