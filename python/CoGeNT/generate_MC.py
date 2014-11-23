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
# Slow parameters
# Using Nicole's simulated stuff
mu = [0.269108,0.747275,0.068146]
sigma = [0.531530,-0.020523]

# Fast parameters
#Using Nicole's simulated stuff
fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
fast_num_rel_k = [-0.261322,5.553102,-5.9144]

mu0 = [0.374145,0.628990,-1.369876]
sigma0 = [1.383249,0.495044,0.263360]
################################################################################

################################################################################
################################################################################
def write_output_file(energy,time_stamps,rise_times,file_name):

    #zip(energy,time_stamps,rise_times)
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(energy,time_stamps,rise_times))
        f.close()

################################################################################
################################################################################
def gen_surface_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    max_prob_calculated = -999
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

        if max_prob_calculated<prob:
            print "Max prob to now: %f" % (prob)
            max_prob_calculated = prob

        if max_prob<prob:
            print prob

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1

    write_output_file(energies,days,rise_times,name_of_output_file)

    return energies,days,rise_times


################################################################################
################################################################################
def gen_flat_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

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

        data = [e,t,0,rt_fast,0]

        prob = flat_events(data,pars,lo,hi,subranges=subranges,efficiency=None)

        if max_prob_calculated<prob:
            print "Max prob to now: %f" % (prob)
            max_prob_calculated = prob

        if max_prob<prob:
            print prob

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1

    write_output_file(energies,days,rise_times,name_of_output_file)
                            
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

    npts_to_generate = 5000

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

    max_prob = 11.0
    energies = np.zeros(maxpts)
    days = np.zeros(maxpts)
    rise_times = np.zeros(maxpts)

    npts = 0
    max_prob_calculated = -999
    efficiency = None
    while npts < maxpts:

        e = (1.1*np.random.random(npts_to_generate) + 0.5) # Generate over a smaller range for the L-shell peaks
        t = (max_days)*np.random.random(npts_to_generate)
        rt = (3.0)*np.random.random(npts_to_generate)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

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
        
        if max(prob)>max_prob_calculated:
            print "Max prob to now: %f" % max(prob)
            max_prob_calculated = max(prob)

        if len(prob[prob>max_prob])>0:
            print max_prob,prob[prob>max_prob]

        probtest = max_prob*np.random.random(npts_to_generate) # This is to see whether or not we keep x!

        #if probtest<prob:
        index = probtest<prob
        npts_good = len(index[index==True])
        if len(index)>0:
            #print "-----------"
            #print energies[npts:npts+npts_good] 
            #print e[index]

            final_index = npts+npts_good
            max_to_insert = npts_good
            #print len(energies),final_index,npts
            if final_index>len(energies):
                final_index = len(energies)
                max_to_insert = len(energies)-npts

            #print "final_index: ",final_index
            energies[npts:final_index] = e[index][0:max_to_insert]
            days[npts:final_index] = t[index][0:max_to_insert]
            rise_times[npts:final_index] = rt[index][0:max_to_insert]
            
            npts += npts_good
            #print npts

    write_output_file(energies,days,rise_times,name_of_output_file)
                            
    return energies,days,rise_times


################################################################################
# Read in from a previous fits of results.
results_file = open(sys.argv[1])
results = eval(results_file.readline())
print results
#exit()

#energies,days,rise_times = gen_surface_events(20,365,'MC_files/mc_test_surface.dat',results)
#energies,days,rise_times = gen_flat_events(20,365,'MC_files/mc_test_flat.dat',results)
energies,days,rise_times = gen_cosmogenic_events(1000,1238,'MC_files/mc_test_lshell.dat',results)
#print energies,days,rise_times 

#'''
nbins = 50
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
h = plt.hist(energies,bins=nbins)
plt.subplot(1,3,2)
h = plt.hist(days,bins=nbins)
plt.subplot(1,3,3)
h = plt.hist(rise_times,bins=nbins)

plt.show()
#'''
