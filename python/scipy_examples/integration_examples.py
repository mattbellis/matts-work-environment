import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

lo = -0.5
hi = 1.0

################################################################################
def double_gauss(y,x):

    g0 = stats.norm(loc=0.0,scale=0.8)
    g1 = stats.norm(loc=0.0,scale=0.5)

    return g0.pdf(x)*g1.pdf(y)
################################################################################



g0 = stats.norm(loc=0.0,scale=0.8)
g1 = stats.norm(loc=0.0,scale=0.5)

x0 = np.linspace(lo,hi,1000)
y0 = g0.pdf(x0)

x1 = np.linspace(lo,hi,1000)
y1 = g1.pdf(x1)

g0_int = integrate.simps(y0,x=x0)
g1_int = integrate.simps(y1,x=x1)

print "g0_int: ",g0_int
print "g1_int: ",g1_int

g0_int = integrate.quad(g0.pdf,lo,hi)
g1_int = integrate.quad(g1.pdf,lo,hi)

print "g0_int: ",g0_int
print "g1_int: ",g1_int

gdbl_int = integrate.dblquad(double_gauss,lo,hi,lambda x: lo, lambda x: hi)

print "g0_int*g1_int: ",g0_int[0]*g1_int[0]
print "gdbl_int: ",gdbl_int 

