#import dm_models as dmm
import matplotlib.pylab as plt
import numpy as np
from chris_kelso_code import dRdErSHM
import csv
import numpy as np
from cogent_utilities import *
from cogent_pdfs import surface_events,flat_events
import lichen.pdfs as pdfs

import parameters

################################################################################
################################################################################
def gen_surface_events(maxpts,max_days,name_of_output_file):

    ranges,subranges,nbins = parameters.fitting_parameters(0)
    print ranges
    print subranges

    pars = {}
    pars['k1_surf'] = -0.5125
    pars['k2_surf'] = 0.0806
    pars['t_surf'] = 0.0002
    pars['num_surf'] = 6000

    # Using Nicole's simulated stuff
    mu = [0.269108,0.747275,0.068146]
    sigma = [0.531530,-0.020523]

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    max_prob = 0.65
    energies = []
    days = []
    rise_times = []

    npts = 0
    while npts < maxpts:

        e = (2.5*np.random.random(1) + 0.5) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_slow = rise_time_prob_exp_progression(rt,e,mu,sigma,ranges[2][0],ranges[2][1])

        data = [e,t,0,0,rt_slow]

        prob = surface_events(data,pars,lo,hi,subranges=subranges,efficiency=None)


        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1

    '''
    zip(a,b,c)
    with open(name_of_output_file,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(a,b,c))
    '''
                            
    return energies,days,rise_times


################################################################################
################################################################################
def gen_flat_events(maxpts,max_days,name_of_output_file):

    ranges,subranges,nbins = parameters.fitting_parameters(0)
    print ranges
    print subranges

    #0.921,7.4,2.38

    pars = {}
    pars['e_exp_flat'] = 0.0000001
    pars['t_exp_flat'] = 0.0002
    pars['flat_neutrons_slope'] = 0.921
    pars['flat_neutrons_amp'] = 17.4
    pars['flat_neutrons_offset'] = 2.38
    pars['num_comp'] = 2287.
    pars['num_neutrons'] = 862.
    #pars['num_comp'] = 287.
    #pars['num_neutrons'] = 2862.

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    #Using Nicole's simulated stuff
    fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
    fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
    fast_num_rel_k = [-0.261322,5.553102,-5.9144]

    mu0 = [0.374145,0.628990,-1.369876]
    sigma0 = [1.383249,0.495044,0.263360]

    max_prob = 4.533
    energies = []
    days = []
    rise_times = []

    npts = 0
    max_prob = -999
    while npts < maxpts:

        e = (2.5*np.random.random(1) + 0.5) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])
        #rt_slow = rise_time_prob_exp_progression(rt,e,mu,sigma,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        prob = flat_events(data,pars,lo,hi,subranges=subranges,efficiency=None)
        if max_prob<prob:
            print prob
            max_prob = prob

        #prob = surface_events(data,pars,lo,hi,subranges=subranges,efficiency=None)

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1

    '''
    zip(a,b,c)
    with open(name_of_output_file,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(a,b,c))
    '''
                            
    return energies,days,rise_times



#energies,days,rise_times = gen_surface_events(2000,365,'mc_test.dat')
energies,days,rise_times = gen_flat_events(1000,365,'mc_test.dat')
#print energies,days,rise_times 

nbins = 25
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
h = plt.hist(energies,bins=nbins)
plt.subplot(1,3,2)
h = plt.hist(days,bins=nbins)
plt.subplot(1,3,3)
h = plt.hist(rise_times,bins=nbins)

plt.show()
