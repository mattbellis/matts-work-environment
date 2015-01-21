#!/usr/bin/env python 

from math import *

nBpairs = 470.89
# Assuming 0.28% 
#nBpairs_err =  1.32
# Assuming 0.6% 
nBpairs_err =  2.83

nB_bf = [0.484, 0.484, 0.516, 0.516, 0.516, 0.516, 0.516]
nB_bf_err = [0.006,0.006,0.006,0.006,0.006,0.006]

# initial numbers for signal SP
skim_eff = [0.467, 0.504, 0.553, 0.569, 0.553, 0.569]
skim_eff_err = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

# initial numbers for signal SP
initial_numbers = [22000, 22000, 28000, 28000, 28000, 28000]
final_numbers =   [12387, 11223, 14536, 13371, 15867, 14760]

nmodes = len(initial_numbers)

# Baryon branching fractions
baryon_bf =     [0.050, 0.050, 0.639, 0.639, 0.639, 0.639]
baryon_bf_err = [0.013, 0.013, 0.005, 0.005, 0.005, 0.005]

# Tracking errors
# http://www.slac.stanford.edu/BFROOT/www/Physics/TrackEfficTaskForce/TauEff/R24/TauEff.html
trk_err_per_trk = 0.128/100.0 # percent error
trk_err_pct_l0 = 3.0 * trk_err_per_trk # 3 tracks 
trk_err_pct_lc = 4.0 * trk_err_per_trk # 4 tracks 
trk_pct_err = [trk_err_pct_lc, trk_err_pct_lc, trk_err_pct_l0, trk_err_pct_l0, trk_err_pct_l0, trk_err_pct_l0]

# PID errors
pid_err_p  = 0.010
pid_err_pi = 0.010
pid_err_k  = 0.012
#pid_err_e  = 0.004
#pid_err_mu = 0.007
pid_err_e  = 0.01
pid_err_mu = 0.025

pid_pct_err = []
# Precise
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_k*pid_err_k + pid_err_mu*pid_err_mu))
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_k*pid_err_k + pid_err_e*pid_err_e))
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_mu*pid_err_mu))
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_e*pid_err_e))
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_mu*pid_err_mu))
pid_pct_err.append(sqrt(pid_err_p*pid_err_p + pid_err_pi*pid_err_pi + pid_err_e*pid_err_e))
# Estimate
#pid_pct_err.append(0.025)
#pid_pct_err.append(0.025)
#pid_pct_err.append(0.020)
#pid_pct_err.append(0.020)
#pid_pct_err.append(0.020)
#pid_pct_err.append(0.020)


# Eff calculations
for i in range(0,nmodes):
    n0 = initial_numbers[i]
    n =  final_numbers[i]
    eff = n/float(n0) 
    eff_err = sqrt((eff*(1.0-eff))/n0)

    pre_skim_eff = eff

    eff *= skim_eff[i]

    conv_factor = (nBpairs*2.0*nB_bf[i]) * eff * baryon_bf[i]

    # Calculate all the percent errors. 
    pct_errs = []

    # number of Bs
    pct_errs.append(nBpairs_err/float(nBpairs))
    # B branching fraction
    pct_errs.append(nB_bf_err[i]/float(nB_bf[i]))

    # Efficiency
    pct_errs.append(skim_eff_err[i]/skim_eff[i])
    pct_errs.append(eff_err/eff)

    # Branching fractions
    pct_errs.append(baryon_bf_err[i]/baryon_bf[i])

    # Tracking
    pct_errs.append(trk_pct_err[i])
    # PID?
    pct_errs.append(pid_pct_err[i])

    eff_tot_err =  (eff_err/eff)*(eff_err/eff)
    eff_tot_err += trk_pct_err[i]*trk_pct_err[i]
    eff_tot_err += pid_pct_err[i]*pid_pct_err[i]

    tot_pct_err = 0.0
    for pe in pct_errs:
        tot_pct_err += pe*pe
        #print "%f %f %f %f" % (tot_pct_err, sqrt(tot_pct_err), pe*pe, pe)

    conv_factor_err = sqrt(tot_pct_err)
    #print "conv_factor_err: %f" % (conv_factor_err)

    # Convert back to a number, rather than a percentage
    conv_factor_err *= conv_factor

    output = "%d\ttrk_pct_err: %6.4f\n" % (i, trk_pct_err[i])
    output += " \tpid_pct_err: %6.4f\n" % (pid_pct_err[i])
    output += " \tnBpairs_err: %6.4f\n" % (nBpairs_err/float(nBpairs))
    output += " \tnB_bf_err: %6.4f\n" % (nB_bf_err[i]/float(nB_bf[i]))
    output += " \tbaryon_bf_err: %6.4f\n" % (baryon_bf_err[i]/baryon_bf[i])
    output += "\tpre_skim_eff: %6.4f +/- %6.4f\t\teff: %6.4f +/- %6.4f" % \
            (pre_skim_eff,eff_err, eff,eff*sqrt(eff_tot_err))
    output += "\t\tconv_factor: %6.2f +/- %6.3f (pct_err: %6.3f)" % \
            (conv_factor,conv_factor_err,100*conv_factor_err/conv_factor)

    print output



