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

import sys

################################################################################
################################################################################
def gen_surface_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)
    print ranges
    print subranges

    #pars = {}
    #pars['k1_surf'] = -0.5125
    #pars['k2_surf'] = 0.0806
    #pars['t_surf'] = 0.0002
    #pars['num_surf'] = 6000

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
def gen_flat_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)
    print ranges
    print subranges

    #0.921,7.4,2.38

    #pars = {}
    #pars['e_exp_flat'] = 0.0000001
    #pars['t_exp_flat'] = 0.0002
    #pars['flat_neutrons_slope'] = 0.921
    #pars['flat_neutrons_amp'] = 17.4
    #pars['flat_neutrons_offset'] = 2.38
    #pars['num_comp'] = 2287.
    #pars['num_neutrons'] = 862.
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
    max_prob_calculated = -999
    while npts < maxpts:

        e = (2.5*np.random.random(1) + 0.5) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])
        #rt_slow = rise_time_prob_exp_progression(rt,e,mu,sigma,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        prob = flat_events(data,pars,lo,hi,subranges=subranges,efficiency=None)
        if max_prob_calculated<prob:
            print prob
            max_prob_calculated = prob

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

################################################################################
################################################################################
def gen_cosmogenic_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)
    print ranges
    print subranges

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    ############################################################################
    # l-shell peaks
    ############################################################################
    means = []
    sigmas = []
    numls = []
    decay_constants = []
    
    num_tot = 0

    for i in xrange(11):
        name = "ls_mean%d" % (i)
        means.append(pars[name])
        name = "ls_sigma%d" % (i)
        sigmas.append(pars[name])
        name = "ls_ncalc%d" % (i)
        #numls.append(pars[name]/num_tot) # Normalized this # to number of events.
        numls.append(pars[name])
        num_tot += pars[name]
        name = "ls_dc%d" % (i)
        decay_constants.append(pars[name])

    print means
    print sigmas
    print numls
    print decay_constants
    #Using Nicole's simulated stuff
    fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
    fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
    fast_num_rel_k = [-0.261322,5.553102,-5.9144]

    mu0 = [0.374145,0.628990,-1.369876]
    sigma0 = [1.383249,0.495044,0.263360]

    max_prob = 1.0
    energies = []
    days = []
    rise_times = []

    npts = 0
    max_prob_calculated = -999
    efficiency = None
    while npts < maxpts:

        e = (1.1*np.random.random(1) + 0.5) # Generate over a smaller range for the L-shell peaks
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])
        #rt_slow = rise_time_prob_exp_progression(rt,e,mu,sigma,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        tot_pdf = 0
        for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
            pdf  = pdfs.gauss(e,m,s,lo[0],hi[0],efficiency=efficiency)
            #print e,pdf
            dc = -1.0*dc
            pdf *= pdfs.exp(t,dc,lo[1],hi[1],subranges=subranges[1])
            pdf *= rt_fast # This will be the fast rise times
            pdf *= n
            tot_pdf += pdf


        prob = tot_pdf
        #print e,prob
        
        if max_prob_calculated<prob:
            print prob
            max_prob_calculated = prob

        #prob = surface_events(data,pars,lo,hi,subranges=subranges,efficiency=None)

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            print npts
            npts += 1

    '''
    zip(a,b,c)
    with open(name_of_output_file,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(a,b,c))
    '''
                            
    return energies,days,rise_times


# Read in from a previous fits of results.
results_file = open(sys.argv[1])
results = eval(results_file.readline())
print results
#exit()

#energies,days,rise_times = gen_surface_events(2000,365,'mc_test.dat',results)
#energies,days,rise_times = gen_flat_events(1000,365,'mc_test.dat',results)
energies,days,rise_times = gen_cosmogenic_events(300,1238,'mc_test.dat',results)
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
