import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

lo = -0.5
hi = 1.0

pi = np.pi

################################################################################
def func(x,y):
    #return np.cos(x)*np.cos(x) + np.sin(y)*np.sin(y)
    return np.cos(x)*np.cos(x) * np.sin(y)*np.sin(y)

################################################################################
def double_gauss(y,x):

    g0 = stats.norm(loc=0.0,scale=0.8)
    g1 = stats.norm(loc=0.0,scale=0.5)

    return g0.pdf(x)*g1.pdf(y)
################################################################################

################################################################################
def double_exp(y,x):

    g0 = stats.expon(loc=0.0,scale=1.0)
    g1 = stats.expon(loc=0.0,scale=1.0)

    #g0 = 1.0*x
    #g1 = 1.0*y

    #return g0.pdf(x)*g1.pdf(y)/((g0.pdf(0.5) - g0.pdf(3.2))*(g1.pdf(0.0) - g1.pdf(400)))
    return g0.pdf(0.3*x)*g1.pdf((1.0/270.0)*y)
    #return np.exp(-0.3*x)*np.exp((-1.0/270.0)*y)
    #return g0*g1
    #return 1.0
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

################################################################################
# Exponential example.
################################################################################
print "\n"

pdf_exp = stats.expon(loc=0.0,slope=0.3)
xpts = np.linspace(0.5,3.2,1000)
#ypts = pdf_exp.pdf(xpts)/(pdf_exp.pdf(0.5) - pdf_exp.pdf(3.2))
ypts = np.exp(-0.3*xpts)
norm0 = integrate.simps(ypts,x=xpts)
print "norm0: ",norm0

pdf_exp = stats.expon(loc=0.0,slope=270.0)
xpts = np.linspace(0,400.0,1000)
#ypts = pdf_exp.pdf(xpts)/(pdf_exp.pdf(0.0) - pdf_exp.pdf(400))
ypts = np.exp((-1.0/270.0)*xpts)
norm1 = integrate.simps(ypts,x=xpts)
print "norm1: ",norm1

print "norm0*norm1: ",norm0*norm1

gdbl_int = integrate.dblquad(double_exp,0.5,3.2,lambda x: 0.0, lambda x: 400.0)
print "gdbl_int: ",gdbl_int

# Manual integral
tot_int = 0.0
xw = 0.02
yw = 0.2
for xpts in np.arange(0.5,3.2,xw):
    for ypts in np.arange(0.0,400.0,yw):
        val =  np.exp(-0.3*xpts)
        val *= np.exp((-1.0/270.0)*ypts)
        tot_int += val*xw*yw

print "manual_int: ",tot_int

#print "g0_int*g1_int: ",g0_int[0]*g1_int[0]
#print "gdbl_int: ",gdbl_int 
#print integrate.dblquad(func,-pi/2, pi/2,lambda x:-pi/2,lambda x:pi/2)
