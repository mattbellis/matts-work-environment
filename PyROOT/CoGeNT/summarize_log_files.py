#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

from math import *

################################################################################
# Parse the file name
################################################################################

################################################################################
# main
################################################################################
def main():

    par_names = ['nbkg', 'nsig', 'sig_slope', 'cg_mod_amp', 'cg_mod_phase', 'bkg_mod_amp', 'bkg_mod_phase', 'sig_mod_amp', 'sig_mod_phase']
    info_flags = ['e_lo', 'signal_modulation', 'background_modulation', 'cosmogenic_modulation', 'add_gc', 'gc_flag']

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
                    
                    #print line
                elif 'likelihood:' in line:

                    vals = line.split()

                    nlls.append(float(vals[3]))

                elif 'INFO:' in line:

                    vals = line.split()

                    #print vals
                    file_info[i-1][vals[1]] = float(vals[2])



    #print "NLLS"
    #print nlls
    #print file_info

    x = []
    y = []
    xerr = []
    yerr = []
    count = 1
    for val in values:
        #print val
        for v in val:
            if v=='nsig':
                #x.append(count)
                x.append(count)
                y.append(val[v][0])
                xerr.append(0.0)
                yerr.append(val[v][1])

        count += 1

    nlls_for_summary = [0.0, 0.0, 0.0, 0.0, 0.0]
    values_for_summary = [None, None, None, None, None]
    print "----------_"
    print file_info
    print len(nlls_for_summary)
    print len(nlls)
    print "----------_"
    for i,f in enumerate(file_info):

        if f['signal_modulation']==0 and f['background_modulation']==0 and f['cosmogenic_modulation']==0:
            nlls_for_summary[0] = nlls[i]
            values_for_summary[0] = values[i]
        elif f['signal_modulation']==0 and f['background_modulation']==1 and f['cosmogenic_modulation']==0:
            nlls_for_summary[1] = nlls[i]
            values_for_summary[1] = values[i]
        elif f['signal_modulation']==1 and f['background_modulation']==0 and f['cosmogenic_modulation']==0:
            nlls_for_summary[2] = nlls[i]
            values_for_summary[2] = values[i]
        elif f['signal_modulation']==1 and f['background_modulation']==1 and f['cosmogenic_modulation']==0:
            nlls_for_summary[3] = nlls[i]
            values_for_summary[3] = values[i]
        elif f['signal_modulation']==1 and f['background_modulation']==1 and f['cosmogenic_modulation']==1:
            nlls_for_summary[4] = nlls[i]
            values_for_summary[4] = values[i]
    
    fit_names = ['none','bkg','sig','sig and bkg','sig and bkg and cosmo']
    dofs = [3, 5, 5, 7, 9]
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "\\frame\n{"
    print "\\frametitle{Relative significance of modulation hypothesis}"
    print "\\large"
    print "\\begin{table}"
    caption = "low energy: %2.1f, Gaussian constraint: %d" % (file_info[0]['e_lo'],file_info[0]['add_gc'])
    print "\\caption{%s}" % (caption)
    print "\\begin{tabular}{l c r r r}"
    print "Modulation & dof & $-\\ln\\mathcal{L}$ & $\\Delta -\\ln\\mathcal{L}$ & $\\sqrt{2.0\\times \\Delta -\\ln \\mathcal{L}}$ \\\\"
    print "\\hline"
    for d,name,n in zip(dofs,fit_names,nlls_for_summary):
        difference = n-nlls_for_summary[0]
        significance = sqrt(-2.0*difference)
        print "%s & %d & %5.2f &  %4.2f & %4.2f \\\\" % (name,d,n,difference,significance)
    print "\\end{tabular}"
    print "\\end{table}"
    print "}"
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    output = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "\\frame\n{\n"
    output += "\\frametitle{Relative significance of modulation hypothesis}\n"
    output += "\\begin{table}\n"
    caption = "low energy: %2.1f, Gaussian constraint: %d" % (file_info[0]['e_lo'],file_info[0]['add_gc'])
    output += "\\caption{%s}\n" % (caption)
    output += "\\begin{tabular}{l c c c c}\n"
    output += " & \\multicolumn{4}{c}{Modulation}\\\\ \n"
    output += "Parameter & None & Background & Signal & Both \\\\\n"
    output += "\\hline\n"
    for par in par_names:
        output += "%-20s  " % (par.replace('_','\_'))
        for val in values_for_summary:
            #print val
            if val is not None and par in val:
                output += " & %7.2f $\pm$ %5.2f   " % (val[par][0],val[par][1])
            else:
                output += " &    "
        output += " \\\\ \n"
    output += "\\end{tabular}\n"
    output += "\\end{table}\n"
    output += "}\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    print output

    ############################################################################
    # Plot the data
    ############################################################################
    fig1 = plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k')
    subplots = []
    for i in range(1,2):
        division = 110 + i
        subplots.append(fig1.add_subplot(division))

    plot = plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
    subplots[0].set_xlim(0,5)

    #plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()

