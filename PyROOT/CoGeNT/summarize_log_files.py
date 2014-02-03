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
            if v=='nexp':
                #x.append(count)
                x.append(count)
                y.append(val[v][0])
                xerr.append(0.0)
                yerr.append(val[v][1])

        count += 1

    nlls_for_summary = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    values_for_summary = [None, None, None, None, None, None]
    #print "----------_"
    #print file_info
    #print len(nlls_for_summary)
    #print len(nlls)
    #print "----------_"
    for i,f in enumerate(file_info):

        if f['exponential_modulation']==0 and f['flat_modulation']==0 and f['cosmogenic_modulation']==0:
            nlls_for_summary[0] = nlls[i]
            values_for_summary[0] = values[i]
        elif f['exponential_modulation']==1 and f['flat_modulation']==0 and f['cosmogenic_modulation']==0:
            nlls_for_summary[1] = nlls[i]
            values_for_summary[1] = values[i]
        elif f['exponential_modulation']==0 and f['flat_modulation']==1 and f['cosmogenic_modulation']==0:
            nlls_for_summary[2] = nlls[i]
            values_for_summary[2] = values[i]
        elif f['exponential_modulation']==0 and f['flat_modulation']==0 and f['cosmogenic_modulation']==1:
            nlls_for_summary[3] = nlls[i]
            values_for_summary[3] = values[i]
        elif f['exponential_modulation']==1 and f['flat_modulation']==1 and f['cosmogenic_modulation']==0:
            nlls_for_summary[4] = nlls[i]
            values_for_summary[4] = values[i]
        elif f['exponential_modulation']==1 and f['flat_modulation']==1 and f['cosmogenic_modulation']==1:
            nlls_for_summary[5] = nlls[i]
            values_for_summary[5] = values[i]
    
    fit_names = ['none','exp','flat','cos', 'exp, flat','exp, flat, cos']
    quantities = ["Modulation", "dof", "$-\\ln\\mathcal{L}$", "$\\Delta \\ln\\mathcal{L}$", "$\\sqrt{2\\Delta \\ln \\mathcal{L}}$"]
    dofs = [3, 5, 5, 5, 7, 9]
    output = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "\\frame\n{\n"
    output += "\\frametitle{Relative significance of modulation hypothesis}\n"
    #output += "\\normalsize\n"
    output += "\\begin{table}\n"
    caption = "low energy: %2.1f" % (file_info[0]['e_lo'])
    if file_info[0]['add_gc'] == 1:
        caption += ", cosmogenic systematics added (+11 \#dof)"
    else:
        caption += "\\textcolor{black}{, cosmogenic systematics added (+11 \#dof)}"
    output += "\\caption{%s}\n" % (caption)

    #output += "\\begin{tabular}{l c r r r r r}\n"
    #output += "Modulation & dof & $-\\ln\\mathcal{L}$ & $\\Delta \\ln\\mathcal{L}$ & $\\sqrt{2\\Delta \\ln \\mathcal{L}}$ & $D_{\\rm none}(\\%)$& $D_{\\rm exp}(\\%)$  \\\\\n"
    ############ Don't display the -LL
    output += "\\begin{tabular}{l c r r  r r}\n"
    output += "Modulation & dof & $\\Delta \\ln\\mathcal{L}$ & $\\sqrt{2\\Delta \\ln \\mathcal{L}}$ & $D_{\\rm none}(\\%)$& $D_{\\rm exp}(\\%)$  \\\\\n"

    output += "\\hline\n"
    count = 0
    for d,name,n in zip(dofs,fit_names,nlls_for_summary):
        difference0 = nlls_for_summary[0] - n
        significance0 = 0.0
        if difference0>0:
            significance0 = sqrt(2.0*difference0)
        nested0 = stats.chisqprob(2.0*difference0,d-dofs[0])

        if count==0:
            significance0 = None
            difference0 = None
            nested0 = None

        difference1 = None
        significance1 = None
        nested1 = None
        if count>=4:
            difference1 = nlls_for_summary[1] - n
            significance1 = sqrt(2.0*abs(difference1))
            nested1 = stats.chisqprob(2.0*difference1,d-dofs[1])

        #print nested0*100.0
        #print nested1*100.0

        #output += "%s & %d & %5.1f " % (name,d,n)
        ######### Don't display the -LL
        output += "%s & %d " % (name,d)

        if difference0 is None:
            output += "& " 
        else:
            output += "& %4.1f" % (difference0)

        if significance0 is None:
            output += "& " 
        else:
            output += "& %4.1f" % (significance0)

        if nested0 is None:
            output += "& " 
        else:
            output += "& %3.1f" % (100.0*nested0)

        if nested1 is None:
            output += "& " 
        else:
            output += "& %3.1f" % (100.0*nested1)


        output += " \\\\\n" 

        if count==1:
            output += "\\hline\n"

        count += 1

    output += "\\end{tabular}\n"
    output += "\\end{table}\n"
    output += "}\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"

    print output

    output = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "\\frame\n{\n"
    output += "\\frametitle{Parameter values from different fits}\n"
    output += '\\tiny\n'
    output += "\\begin{table}\n"
    caption = "low energy: %2.1f" % (file_info[0]['e_lo'])
    if file_info[0]['add_gc'] == 1:
        caption += ", cosmogenic systematics added (+11 \#dof)"
    else:
        caption += "\\textcolor{black}{, cosmogenic systematics added (+11 \#dof)}"
    
    output += "\\caption{%s}\n" % (caption)
    output += "\\begin{tabular}{l c c c c c c }\n"
    
    for j in xrange(2):
        start = 0; stop = 6
        if j==1:
            start = 6; stop = 9
            output += " & & & "
        for i,p in enumerate(par_names_for_table):
            if i>=start and i<stop:
                output += " & %s" % p
        output += " \\\\\n"
        
        output += "\\hline\n"
        count = 0
        for name,val in zip(fit_names,values_for_summary):
            output += "%-20s  " % (name)
            if j==1:
                output += " & & & "
            for i,par in enumerate(par_names):
                if i>=start and i<stop:
                    if val is not None and par in val:
                        output += " & %8.2f $\pm$ %5.2f   " % (val[par][0],val[par][1])
                    else:
                        output += " &    "
            output += " \\\\ \n"
            count += 1

        if j==0:
            output += "\\hline\n"

    output += "\\end{tabular}\n"
    output += "\\end{table}\n"
    output += "}\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    output += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    print output


    ############################################################################
    # Run test of Gaussian constraints
    ############################################################################
    '''
    for vals in values:
        for v in vals:
            num = vals[v][0]
            uncert = vals[v][1]
            test_val = uncert*uncert/num
            print "%-20s & %6.2f & %6.2f & %6.2f \\\\" % (v,num,uncert,test_val)
    '''

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

