import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime,timedelta

import scipy.integrate as integrate

import parameters
from cogent_utilities import *
from cogent_pdfs import *
from fitting_utilities import *
from lichen.plotting_utilities import *
from plotting_utilities import plot_wimp_er
from plotting_utilities import plot_wimp_day

import lichen.lichen as lch

import iminuit as minuit

import argparse

pi = np.pi
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

np.random.seed(200)

yearly_mod = 2*pi/365.0

ranges,subranges,nbins = parameters.fitting_parameters(0)
nevents = 100000
# Gen random data
e = (ranges[0][1]-ranges[0][0])*np.random.random(nevents) + ranges[0][0]
t = (ranges[1][1]-ranges[1][0])*np.random.random(nevents) + ranges[1][0]
rt = (ranges[2][1]-ranges[2][0])*np.random.random(nevents) + ranges[2][0]

data = [e.copy(),t.copy(),rt.copy()]
print len(data[0])

data = cut_events_outside_range(data,ranges)
data = cut_events_outside_subrange(data,subranges[1],data_index=1)
print len(data[0])

plt.figure()
lch.hist_2D(data[0],data[1],xbins=100,ybins=100,xrange=(ranges[0][0],ranges[0][1]),yrange=(ranges[1][0],ranges[1][1]))

print "Precalculating the slow rise times........"
mu = [0.945067,0.646431,0.353891]
sigma =  [11.765617,94.854276,0.513464]
rt_slow = rise_time_prob_exp_progression(data[2],data[0],mu,sigma,ranges[2][0],ranges[2][1])
data.append(rt_slow)
data.append(rt_slow)
print "Finished with precalcuating rise times........."

pars = {}
pars['e_surf'] = 0.777
pars['t_surf'] = 0.000001
pars['num_surf'] = 50000
pdf = surface_events(data,pars,[ranges[0][0],ranges[1][0]],[ranges[0][1],ranges[1][1]],subranges=subranges,efficiency=None)

plt.figure()
h,x,y,xerr,yerr = lch.hist_err(data[2],bins=50,weights=pdf)
plt.plot(x,y,'b-')
print sum(y)

plt.figure()
h,x,y,xerr,yerr = lch.hist_err(data[0],bins=50,weights=pdf)
plt.plot(x,y,'b-')
print sum(y)

plt.figure()
h,x,y,xerr,yerr = lch.hist_err(data[1],bins=50,weights=pdf)
plt.plot(x,y,'b-')
print sum(y)

plt.show()
