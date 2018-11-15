import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import approx_fprime,fmin_bfgs

from lichen.fit import Parameter,get_numbers,reset_parameters,pois,errfunc,pretty_print_parameters,get_values_and_bounds,fit_emlm

import numpy as np

np.random.seed(0)

################################################################################
def signal(pars, x, frange=None):

    mean = pars["signal"]["mean"].value
    sigma = pars["signal"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################
################################################################################
def signal2(pars, x, frange=None):

    mean = pars["signal2"]["mean"].value
    sigma = pars["signal2"]["sigma"].value

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
    nsig2 = pars["signal2"]["number"].value
    nbkg = pars["bkg"]["number"].value

    ntot = float(nsig + nsig2 + nbkg)

    sig = signal(pars,x,frange=frange)
    sig2 = signal2(pars,x,frange=frange)
    bkg = background(x,frange=frange)

    totpdf = (nsig/ntot)*sig + (nsig2/ntot)*sig2 +  (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Set up your parameters
################################################################################
pars = {}
pars["signal"] = {"number":Parameter(1000,(0,2000)), "mean":Parameter(5.0,(0.1,6.0)), "sigma":Parameter(0.5,(0.1,1.0))}
pars["signal2"] = {"number":Parameter(500,(0,2000)), "mean":Parameter(7.0,(6,10.0)),  "sigma":Parameter(0.5,(0.1,1.0))}
pars["bkg"] = {"number":Parameter(1000,(0,2000))}

################################################################################

################################################################################
# Generate some fake data
################################################################################
data = stats.norm(pars["signal"]["mean"].value,pars["signal"]["sigma"].value).rvs(size=1000).tolist()
data += stats.norm(pars["signal2"]["mean"].value,pars["signal2"]["sigma"].value).rvs(size=500).tolist()
data += (10*np.random.random(1000)).tolist()

initvals,finalvals = fit_emlm(pdf,pars,data)
print("Done with fit!")
pretty_print_parameters(pars)

af = approx_fprime(finalvals,errfunc,1e-8,data, data, [], pars, pdf)
print(af)

# https://gist.github.com/jgomezdans/3144636
def hessian ( x0, calculate_cost_function, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    f1 = approx_fprime( x0, calculate_cost_function, epsilon, *args)

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in range( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime( x0, calculate_cost_function, epsilon, *args) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian

hmat = hessian(finalvals,errfunc,1e-3,False,data, data, [], pars, pdf)
#print(hmat)
#print()
invhmat = np.linalg.inv(hmat)
#print(invhmat)
for i in range(len(finalvals)):
    print(np.sqrt(invhmat[i][i]))
print()

'''
# Maybe for uncertainties, call something that calculates the Hessian?
retvals = fmin_bfgs(errfunc, finalvals, args=(data, data, [], pars,pdf), full_output=True,epsilon=1e-5)
print(finalvals)
print(retvals[0])
invh = retvals[3]
npars = len(finalvals)
print(invh)
print(npars)
for i in range(npars):
    print(np.sqrt(invh[i][i]))
'''



################################################################################
# Plot the results!
################################################################################

xpts = np.linspace(0,10,1000)

plt.figure()
binwidth=(10/100)
plt.hist(data,bins=100,range=(0,10))

ysig = pars['signal']['number'].value*signal(pars,xpts) * binwidth
plt.plot(xpts,ysig,linewidth=3)

ysig2 = pars['signal2']['number'].value*signal2(pars,xpts) * binwidth
plt.plot(xpts,ysig2,linewidth=3)

ybkg = pars['bkg']['number'].value*np.ones(len(xpts))/(10-0) * binwidth
plt.plot(xpts,ybkg,linewidth=3)

##plt.plot(xpts,ybkg + ysig,linewidth=3)
ntot = sum(get_numbers(pars))
ytot = ntot*pdf(pars,xpts,frange=(0,10)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
