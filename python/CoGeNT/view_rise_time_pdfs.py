import numpy
from mayavi.mlab import *
from mayavi import mlab

from cogent_utilities import *

from scipy import integrate


# Rise time and then energy
x, y = numpy.mgrid[0.:6.0:0.05, 0.5:3.2:0.05]
#x, y = numpy.mgrid[0.:3.0:0.1, 0.5:0.6:0.05]
print len(x[0])
print len(y[0])
print x
print y
rt_fast = x.copy()
rt_slow = x.copy()

# Parameters for the exponential form for the narrow fast peak.
mu0 =  [1.016749,0.786867,-1.203125]
sigma0 =  [2.372789,1.140669,0.262251]
# The entries for the relationship between the broad and narrow peak.
fast_mean_rel_k = [0.649640,-1.655929,-0.069965]
fast_sigma_rel_k = [-3.667547,0.000256,-0.364826]
fast_num_rel_k =  [-2.831665,0.023649,1.144240]

for i in range(len(x)):
    rt_fast[i] = rise_time_prob_fast_exp_dist(x[i],y[i],mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,0,6.0)

#'''
for i in range(30):
    xpts = np.linspace(0,6,100)
    energies = 0.5+0.1*i*np.ones(len(xpts))
    yrt = rise_time_prob_fast_exp_dist(xpts,energies,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,0,6.0)
    norm = integrate.simps(yrt,x=xpts)
    print energies[0],norm
#'''


print rt_fast

# Parameters for the exponential form for the slow peak.
# THESE WERE WHAT WAS IN THERE
mu = [0.945067,0.646431,0.353891]
sigma =  [11.765617,94.854276,0.513464]

#mu =  [1.269251,0.802876,0.379299]
#sigma = [0.539896,-0.011620] # THIS IS FOR A LINEAR FUNC.
for i in range(len(x)):
    rt_slow[i] = rise_time_prob_exp_progression(x[i],y[i],mu,sigma,0,6.0)


norm2d_fast = integrate.dblquad(rise_time_prob_fast_exp_dist,0.5,3.2,lambda x:0,lambda x:6.0,args=(mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,0,6.0),epsabs=0.1)

norm2d_slow = integrate.dblquad(rise_time_prob_exp_progression,0.5,3.2,lambda x:0,lambda x:6.0,args=(mu,sigma,0,6.0),epsabs=0.1)

print "norm2d_fast: %f %f" % (norm2d_fast[0],norm2d_fast[1])
print "norm2d_slow: %f %f" % (norm2d_slow[0],norm2d_slow[1])

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1),size=(900,600))
s = mlab.surf(x, y, rt_fast)
mlab.axes(s)
#ranges=(0, 1, 0, 1, 0, 1), xlabel='', ylabel='',
#zlabel='Probability',
#x_axis_visibility=False, z_axis_visibility=False))

mlab.figure(2, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1),size=(900,600))
s = mlab.surf(x, y, rt_slow)
mlab.axes(s)


mlab.show()

