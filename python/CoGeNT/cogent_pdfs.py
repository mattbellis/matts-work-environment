import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

#from RTMinuit import *

import chris_kelso_code as dmm

import lichen.pdfs as pdfs

tc_SHM = dmm.tc(np.zeros(3))
AGe = 72.6

dblqtol = 1.0

################################################################################
# Cosmogenic data
################################################################################
def lshell_data(day_ranges):
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
        
        ndecays = 0
        for dr in day_ranges:
            #ndecays += num_tot_decays[i]*(1.0-np.exp((dr[1]-1)*decay_constants[i]))
            ndecays += num_tot_decays[i]*(np.exp((dr[0]-1)*decay_constants[i])-np.exp((dr[1]-1)*decay_constants[i]))

        num_decays_in_dataset = np.append(num_decays_in_dataset,ndecays)

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

################################################################################
# WIMP signal
################################################################################
def wimp(org_day,x,AGe,mDM,sigma_n,efficiency=None,model='shm'):

    if not (model=='shm' or model=='stream' or model=='debris'):
        print "Not correct model for plotting WIMP PDF!"
        print "Model: ",model
        exit(-1)

    # For debris flow. (340 m/s)
    vDeb1 = 340

    y = (org_day+338)%365.0
    #y = org_day
    xkeVr = dmm.quench_keVee_to_keVr(x)
    #xkeVr = x

    if model=='shm':
        dR = dmm.dRdErSHM(xkeVr,y,AGe,mDM,sigma_n)
    elif model=='debris':
        dR = dmm.dRdErDebris(xkeVr,y,AGe,mDM,vDeb1,sigma_n)
    elif model=='stream':
        #The Sagitarius stream may intersect the solar system
        vSag=300
        v0Sag=100
        vSagHat = np.array([0,0.233,-0.970])
        vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
        streamVel = vSagVec
        streamVelWidth = v0Sag
        dR = dmm.dRdErStream(xkeVr,y,AGe,streamVel,streamVelWidth,mDM,sigma_n)

    eff = 1.0
    if efficiency!=None:
        eff = efficiency(x)

    dR *= eff
#
    # Need this because we've converted from one function to another.
    # dR/dEee
    dR *= ((5.0**(1.0/1.12))/1.12)*(x**(-0.12/1.12))


    return dR


################################################################################
# CoGeNT fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    tot_pdf = np.zeros(len(data[0]))
    pdf = None
    num_wimps = 0

    x = data[0]
    y = data[1]

    xlo = params_dict['var_e']['limits'][0]
    xhi = params_dict['var_e']['limits'][1]
    ylo = params_dict['var_t']['limits'][0]
    yhi = params_dict['var_t']['limits'][1]

    tot_pdf = np.zeros(len(x))

    max_val = 0.86786
    threshold = 0.345
    sigmoid_sigma = 0.241

    efficiency = lambda x: sigmoid(x,threshold,sigmoid_sigma,max_val)
    #efficiency = lambda x: 1.0

    #subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]
    #subranges = [[],[[1,459]]]

    if flag==5 or flag==6: # MC
        subranges = [[],[[1,917]]]

    wimp_model = None
    
    ############################################################################
    # Set up the fit
    ############################################################################
    num_exp0 = p[pn.index('num_exp0')]
    num_flat = p[pn.index('num_flat')]
    e_exp1 = p[pn.index('e_exp1')]
    num_exp1 = p[pn.index('num_exp1')]

    if flag==0 or flag==1 or flag==5:
        e_exp0 = p[pn.index('e_exp0')]

    if flag==1:
        wmod_freq = p[pn.index('wmod_freq')]
        wmod_phase = p[pn.index('wmod_phase')]
        wmod_amp = p[pn.index('wmod_amp')]
        wmod_offst = p[pn.index('wmod_offst')]

    elif flag==2 or flag==3 or flag==4 or flag==6:
        #mDM = 7.0
        mDM = p[pn.index('mDM')]
        sigma_n = p[pn.index('sigma_n')]
        #loE = dmm.quench_keVee_to_keVr(0.5)
        #hiE = dmm.quench_keVee_to_keVr(3.2)
        loE = 0.5
        hiE = 3.2
        if flag==2:
            wimp_model = 'shm'
        elif flag==3:
            wimp_model = 'debris'
        elif flag==4:
            wimp_model = 'stream'
        elif flag==6:
            wimp_model = 'shm'

    ############################################################################
    # Normalize numbers.
    ############################################################################
    num_tot = 0.0
    for name in pn:
        if flag==0 or flag==1:
            if 'num_' in name or 'ncalc' in name:
                num_tot += p[pn.index(name)]
        elif flag==2 or flag==3 or flag==4:
            if 'num_flat' in name or 'num_exp1' in name or 'ncalc' in name:
                num_tot += p[pn.index(name)]
    # MC
    if flag==5:
        num_tot += p[pn.index('num_exp0')]
        num_tot += p[pn.index('num_flat')]

    # MC
    if flag==6:
        num_tot += p[pn.index('num_flat')]

    if flag==2 or flag==3 or flag==4 or flag==6:
        num_wimps = 0
        for sr in subranges[1]:
            num_wimps += integrate.dblquad(wimp,loE,hiE,lambda x: sr[0],lambda x:sr[1],args=(AGe,mDM,sigma_n,efficiency,wimp_model),epsabs=dblqtol)[0]*(0.333)
        num_tot += num_wimps

    print "fitfunc num_tot: %12.3f" % (num_tot)
    ############################################################################
    # Start building the pdfs
    ############################################################################

    ############################################################################
    # l-shell peaks
    ############################################################################
    means = []
    sigmas = []
    numls = []
    decay_constants = []

    for i in xrange(11):
        name = "ls_mean%d" % (i)
        means.append(p[pn.index(name)])
        name = "ls_sigma%d" % (i)
        sigmas.append(p[pn.index(name)])
        name = "ls_ncalc%d" % (i)
        numls.append(p[pn.index(name)]/num_tot) # Normalized this
                                                # to number of events.
        name = "ls_dc%d" % (i)
        decay_constants.append(p[pn.index(name)])

    for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
        pdf  = pdfs.gauss(x,m,s,xlo,xhi,efficiency=efficiency)
        dc = -1.0*dc
        pdf *= pdfs.exp(y,dc,ylo,yhi,subranges=subranges[1])
        pdf *= n
        if flag!=5 and flag!=6:
            tot_pdf += pdf

    ############################################################################
    # Normalize the number of events to the total.
    ############################################################################
    if flag==0 or flag==1 or flag==5:
        num_exp0 /= num_tot

    num_exp1 /= num_tot
    num_flat /= num_tot

    print " -------------------------------------- "
    print x[0:4]
    print y[0:4]
    ########################################################################
    # Wimp-like signal
    ########################################################################
    if flag==0 or flag==5:
        pdf  = pdfs.exp(x,e_exp0,xlo,xhi,efficiency=efficiency)
        pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
        pdf *= num_exp0
        print "exp0 pdf: ",pdf[0:4]
    elif flag==1:
        pdf  = pdfs.exp(x,e_exp0,xlo,xhi,efficiency=efficiency)
        pdf *= pdfs.cos(y,wmod_freq,wmod_phase,wmod_amp,wmod_offst,ylo,yhi,subranges=subranges[1])
        pdf *= num_exp0
    elif flag==2 or flag==3 or flag==4 or flag==6:
        print "num_wimps mDM: %12.3f %12.3f %12.3e" % (num_wimps,mDM,sigma_n)
        wimp_norm = num_wimps
        #print "wimp_norm: ",wimp_norm
        pdf = wimp(y,x,AGe,mDM,sigma_n,efficiency=efficiency,model=wimp_model)/(1.0*num_tot)
        pdf *= (0.333) # Active volume of CoGeNT
        print "wimp pdf: ",pdf[0:4]*(num_tot)/num_wimps
        print "wimp pdf: ",pdf[0:4]

    #print "wimp pdf: ",pdf[0:4]
    tot_pdf += pdf

    ############################################################################
    # Second exponential in energy
    ############################################################################
    pdf  = pdfs.exp(x,e_exp1,xlo,xhi,efficiency=efficiency)
    pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
    pdf *= num_exp1
    #print "exp1 pdf: ",pdf[0:4]
    if flag!=5 and flag!=6:
        tot_pdf += pdf

    ############################################################################
    # Flat term
    ############################################################################
    pdf  = pdfs.poly(x,[],xlo,xhi,efficiency=efficiency)
    pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
    pdf *= num_flat
    print "flat pdf: ",pdf[0:4]/num_flat
    print "flat pdf: ",pdf[0:4]
    tot_pdf += pdf

    print "%f %f %f" % (num_tot,num_wimps/num_tot,num_flat)

    return tot_pdf
################################################################################


