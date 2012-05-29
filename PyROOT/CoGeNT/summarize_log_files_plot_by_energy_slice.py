#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats.stats as stats

from math import *

################################################################################
# Parse the file name
################################################################################

################################################################################
# main
################################################################################
def main():

    par_names = ['nflat', 'nexp', 'exp_slope', 'flat_mod_amp', 'exp_mod_amp', 'cg_mod_amp', 'flat_mod_phase', 'exp_mod_phase', 'cg_mod_phase']
    par_names_for_table = ['$N_{flat}$', '$N_{exp}$', '$\\alpha$', '$A_{flat}$', '$A_{exp}$', '$A_{cg}$', '$\phi_{flat}$', '$\phi_{exp}$', '$\phi_{cg}$']
    info_flags = ['e_lo', 'exponential_modulation', 'flat_modulation', 'cosmogenic_modulation', 'add_gc', 'gc_flag']

    values = []
    nlls = []
    file_info = []
    for i,file_name in enumerate(sys.argv):

        #print file_name
        #print len(nlls)

        if i>0:

            values.append({})
            file_info.append({})

            infile = open(file_name)

            for line in infile:
                
                if 'none' in line:

                    vals = line.split()

                    name = vals[0]

                    #par_names.index(name)

                    values[i-1][name] = [float(vals[2]),float(vals[4])]
                    #nlls.append(float(vals[3]))
                    
                    #print line
                elif 'likelihood:' in line:

                    vals = line.split()

                    values[i-1]['nll'] = float(vals[3])

                elif 'INFO:' in line:

                    vals = line.split()

                    #print vals
                    file_info[i-1][vals[1]] = float(vals[2])
                    values[i-1][vals[1]] = float(vals[2])



    #print "NLLS"
    #print nlls
    #print file_info
    for f in file_info:
        print f
    
    x = []
    y = []
    xerr = []
    yerr = []
    #print values
    for v in values:
        #print v
        if v['flat_modulation']==0:
            elo = v['e_lo']
            ehi = v['e_hi']
            nll0 = v['nll']
            for vv in values:
                #print vv
                if vv['flat_modulation']==1.0 and elo == vv['e_lo'] and ehi == vv['e_hi']:
                    nll1 = vv['nll']

                    width = ehi-elo
                    width_cutoff = 0.8
                    if width>width_cutoff-0.05 and width<width_cutoff+0.05:
                        Dpct = 100.0*stats.chisqprob(2*abs(nll0-nll1),2)
                        x.append((ehi+elo)/2.0)
                        xerr.append((ehi-elo)/2.0)
                        y.append(Dpct)
                        yerr.append(0.001)
                        print "%4.1f %4.1f %f %f %f" % (elo,ehi,nll0,nll1,Dpct)
            

    ############################################################################
    # Plot the data
    ############################################################################
    fig1 = plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k')
    subplots = []
    for i in range(1,2):
        division = 110 + i
        subplots.append(fig1.add_subplot(division))

    plot = plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
    subplots[0].set_xlim(0,3.5)
    subplots[0].set_ylim(0,100.0)

    #exit()
    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()

