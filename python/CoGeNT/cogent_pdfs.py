import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

from RTMinuit import *


################################################################################
# Cosmogenic data
################################################################################
def lshell_data(days):
    lshell_data_dict = {
        "As 73": [125.45,33.479,0.11000,13.799,12.741,1.4143,0.077656,80.000,0.0000,11.10,80.0],
        "Ge 68": [6070.7,1.3508,0.11400,692.06,638.98,1.2977,0.077008,271.00,0.0000,10.37,271.0],
        "Ga 68": [520.15,5.1139,0.11000,57.217,52.828,1.1936,0.076426,271.00,0.0000,9.66,271.0],
        "Zn 65": [2117.8,2.2287,0.10800,228.72,211.18,1.0961,0.075877,244.00,0.0058000,8.98,244.0],
        "Ni 56": [16.200,23.457,0.10200,1.6524,1.5257,0.92560,0.074906,5.9000,0.39000,7.71,5.9],
        "Co 56/58": [100.25,8.0,0.10200,10.226,9.4412,0.84610,0.074449,71.000,0.78600,7.11,77.0],
        "Co 57": [27.500,8.0,0.10200,2.8050,2.5899,0.84610,0.074449,271.00,0.78600,7.11,271.0],
        "Fe 55": [459.20,11.629,0.10600,48.675,44.942,0.76900,0.074003,996.00,0.96000,6.54,996.45],
        "Mn 54": [223.90,9.3345,0.10200,22.838,21.086,0.69460,0.073570,312.00,1.0000,5.99,312.0],
        "Cr 51": [31.500,15.238,0.10100,3.1815,2.9375,0.62820,0.073182,28.000,1.0000,5.46,28.0],
        "V 49": [161.46,12.263,0.10000,16.146,14.908,0.56370,0.072803,330.00,1.0000,4.97,330.0],
        }

    means = np.array([])
    sigmas = np.array([])
    num_tot_decays = np.array([])
    decay_constants = np.array([])
    num_decays_in_dataset = np.array([])

    for i,p in enumerate(lshell_data_dict):

        means = np.append(means,lshell_data_dict[p][5])
        sigmas = np.append(sigmas,lshell_data_dict[p][6])

        half_life = lshell_data_dict[p][7]
        decay_constants = np.append(decay_constants,-1.0*np.log(2.0)/half_life)

        num_tot_decays = np.append(num_tot_decays,lshell_data_dict[p][4])
        # *Before* the efficiency?
        #num_tot_decays = np.append(num_tot_decays,lshell_data_dict[p][3])
        
        num_decays_in_dataset = np.append(num_decays_in_dataset,num_tot_decays[i]*(1.0-np.exp(days*decay_constants[i])))

    return means,sigmas,num_tot_decays,num_decays_in_dataset,decay_constants



################################################################################
# Cosmogenic peaks
################################################################################
def lshell_peaks(means,sigmas,numbers):

    npeaks = len(means)

    pdfs = []

    for mean,sigma,number in zip(means,sigmas,numbers):
        pdf = sp.stats.norm(loc=mean,scale=sigma)
        pdfs.append(pdf)

    return pdfs

################################################################################
# Sigmoid function.
################################################################################
def sigmoid(x,thresh,sigma,max_val):

    ret = max_val / (1.0 + np.exp(-(x-thresh)/(thresh*sigma)))

    return ret


