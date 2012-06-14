"""
Extended maximum likelihood method
"""
import minuit
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
import lichen.lichen as lch

# Generate the data.
mean = 5.0
sigma = 0.5
data = np.random.normal(mean,sigma,100)

mc = 8*np.random.random(10000)

fig0 = plt.figure(figsize=(10,10))
ax0 = fig0.add_subplot(2,1,1)
ax1 = fig0.add_subplot(2,1,2)

hdata = lch.hist_err(data,bins=50,range=(0,8),axes=ax0)
hmc = lch.hist_err(mc,bins=50,range=(0,8),axes=ax1)


################################################################################
def f(x, y):
    return ((x-2) / 3)**2 + y**2 + y**4
################################################################################

################################################################################
def pois(mu, k):
    ret = -mu + k*np.log(mu)
    return ret

################################################################################

################################################################################
def fitfunc(p,x):
    mean = p[0]
    sigma = p[1]
    function = stats.norm(loc=mean,scale=sigma)

    ret = function.pdf(x)

    return ret
################################################################################
    
    
################################################################################
#def errfunc(p0,p1,n):
def errfunc(pars):
    print pars
    p0 = pars[0]
    p1 = pars[1]
    n = pars[2]
    norm_func = (fitfunc([p0,p1], mc)).sum()/len(mc)
    ret = 0.0
    if norm_func==0:
        norm_func = 1000000.0
    
    ret = (-np.log(fitfunc([p0,p1], data) / norm_func).sum()) - pois(n,len(data))
    return ret
################################################################################



#m = minuit.Minuit(f, x=10, y=10)
#p = [0.0,0.0]
#m = minuit.Minuit(errfunc, p0=5.0, p1=0.5,n=50)
pars = [4.0,0.1,50]
#m = minuit.Minuit(errfunc, pars)
m = minuit.Minuit(errfunc)
print m.parameters
print m.values
#m.values[pars[0]] = 4.0
#m.values[pars[1]] = 0.5
#m.values[pars[2]] = 50.0
#m.values['p0'] = 4.0
#m.values['p1'] = 0.5
#m.values['n'] = 50.0
m.values[pars] = pars
print m.values
print m.fixed
#m.parameters[0] = 5.0
#m.parameters[1] = 0.5
#m.fixed['p0'] = True
print m.fixed
print m.args

#m.printMode = 1

m.migrad()
print m.fval, m.ncalls, m.edm
print "Values: "
print m.values["p0"], m.values["p1"], m.values["n"]

m.hesse()
print "Errors: "
print m.errors

print "Covariance"
print m.covariance
print "Correlation matrix"
print m.matrix(correlation=True)

m.minos('p1',1.0)
m.minos('p1',-1.0)
#m.minos()
print "Minos errors: "
print m.merrors


print "Contour: " 
print m.contour("p0", "p1", 1.)[:3]

print "Scan: " 
#scan = m.scan(("p0", 5, 1, 10), ("p1", 5, 0.01, 10), corners=True)
scan = m.scan(("p0", 5, 3, 7), ("p1", 5, 0.10, 0.9))
print scan[0]
print scan[1]

plt.show()
