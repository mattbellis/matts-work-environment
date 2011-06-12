#!/usr/bin/env python

from math import *
from optparse import OptionParser
import sys
import numpy
import random

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MaxNLocator
from matplotlib import rc


from bootstrapping import *

################################################################################
################################################################################
def main(argv):

    #### Command line variables ####
    parser = OptionParser()
    parser.add_option("--ntrials", dest="ntrials", default=1000, help='Number of\
                        trials to run in bootstrapping approach.')
    parser.add_option("--nn-lo", dest="nn_lo", default=0.7, help='Low edge of \
                        neural net output on which to cut.')
    parser.add_option("--tag", dest="tag", default='default', help='Tag for \
                        output file.')
    parser.add_option("--interval", dest="interval", default=0.95, \
                      help='Confidence interval over which to calculate the \
                      error.')
    parser.add_option("--batch", dest="batch", action='store_true', 
            default=False, help='Run in batch mode')

    (options, args) = parser.parse_args()

    xlo = [5.2,-0.2,float(options.nn_lo)]
    xhi = [5.3, 0.2,1.1]

    ntrials = int(options.ntrials)
    interval = float(options.interval)

    ################################################################################
    # Make a figure on which to plot stuff.
    figs = []
    for i in range(0,8):
        figs.append(plt.figure(figsize=(10, 2.5), dpi=100, facecolor='w', edgecolor='k'))
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ################################################################################
    subplots = []
    subplots_2d = []
    subplots_nn = []
    subplots_vars = []
    for i in range(1,7):
        division = 131 + (i-1)%3
        subplots.append(figs[0+(i-1)/3].add_subplot(division))
        subplots_2d.append(figs[2+(i-1)/3].add_subplot(division))
        subplots_nn.append(figs[4+(i-1)/3].add_subplot(division))
        subplots_vars.append(figs[6+(i-1)/3].add_subplot(division))
    ################################################################################
    # For Latex stuff
    rc('text', usetex=True)
    rc('font', family='serif')
    ################################################################################

    # Calculate the interval indices over which we will calculate the error.
    # Interval indices
    i_lo = int(ntrials*(0.50 - (interval/2.0)))
    i_hi = int(ntrials*(0.50 + (interval/2.0)))

    print i_lo
    print i_hi

    ############################################################################
    # Open the file
    infilename = args[0]
    infile = open(infilename)

    # Get the tag for the output files
    tag = infilename.split('/')[-1].split('.txt')[0]

    #tag += "_ntrials%d_nn%d" % (ntrials, int(float(options.nn_lo)*100))
    tag += "_ntrials%d" % (ntrials)

    outfilename = "matplotlib_textFiles/%s.txt" % (tag)
    outfile = open(outfilename,"w+")
    text_out_name = "correlation_coefficient_outputs/%s.txt" % (tag)
    text_out_file = open(text_out_name,"w")


    # Get all the values
    x = [[], [], []]
    nn = []
    n = 0
    for line in infile:
        vals = line.split()

        # Check that the event is within our ranges
        good_event = True
        for i,v in enumerate(vals):
            z = float(v)

            if i==2:
                nn.append(z)

            if z<xlo[i] or z>xhi[i]:
                good_event = False

        if good_event:
            x[0].append(float(vals[0]))
            x[1].append(float(vals[1]))
            x[2].append(float(vals[2]))

            n+=1

        if n==2000:
            break

    nentries = len(x[0])
    output = "Original sample size: %d" % (nentries)
    print output
    outfile.write(output)
    sample_size = nentries

    ############################################################################
    # Plot the NN
    for i in range(0,6):
        subplots_nn[i].hist(nn, bins=130, range=(0.4,1.05), normed=0, \
                facecolor='yellow', alpha=0.99, histtype='stepfilled')
        subplots_nn[i].hist(x[2], bins=130, range=(0.4,1.05), normed=0, \
                facecolor='blue', alpha=0.99, histtype='stepfilled')

    ############################################################################
    # Grab the means and sigmas
    mean = []
    sigma = []
    for i in range(0,3):
        mean.append(numpy.mean(x[i]))
        sigma.append(numpy.std(x[i]))

    ############################################################################
    # Calculate correlation coefficients over the whole area
    cc = []
    cc_all_trials = []
    index = []
    cc_intervals = [[],[]]
    for i in range(0,3):
        for j in range(i+1,3):
            cc_trials, c  = calc_coeff_and_subsamples(x[i],x[j],\
                            ntrials,sample_size)
            cc.append([c, numpy.mean(cc_trials), numpy.std(cc_trials)])
            cc_all_trials.append(cc_trials)
            index.append([i,j])
            cc_trials.sort()
            #print cc_sort
            print "%f %f" % (cc_trials[i_lo], cc_trials[i_hi])
            cc_intervals[0].append(cc_trials[i_lo])
            cc_intervals[1].append(cc_trials[i_hi])

    print index

    ############################################################################
    for i in range(0,3):
        subplots_2d[i].scatter(x[0],x[1],s=2)
        if i==0:
            subplots_vars[i].scatter(x[0],x[1],s=2)
        elif i==1:
            subplots_vars[i].scatter(x[0],x[2],s=2)
        elif i==2:
            subplots_vars[i].scatter(x[1],x[2],s=2)

    ############################################################################
    # Calculate correlation coefficients over subsections.
    xsub = [[], [], []]
    ysub = [[], [], []]

    xplt = [[], [], []]
    yplt = [[], [], []]

    for i in range(0,nentries):
        mes = x[0][i] # mES
        dle = x[1][i] # NN
        nno = x[2][i] # NN

        # Range 1
        if dle>-0.075 and dle<0.075:
            xsub[0].append(mes)
            ysub[0].append(nno)

            xplt[0].append(mes)
            yplt[0].append(dle)

        # Range 2
        if mes>5.27 and dle>-0.20 and dle<0.075:
            xsub[1].append(dle)
            ysub[1].append(nno)

            xplt[1].append(mes)
            yplt[1].append(dle)

        # Range 3
        if mes>5.27 and dle>-0.075 and dle<0.20:
            xsub[2].append(dle)
            ysub[2].append(nno)

            xplt[2].append(mes)
            yplt[2].append(dle)

    for i in range(0,3):
        cc_trials, c  = calc_coeff_and_subsamples(xsub[i],ysub[i],ntrials,\
                                 sample_size)
        cc.append([c, numpy.mean(cc_trials), numpy.std(cc_trials)])
        cc_all_trials.append(cc_trials)
        cc_trials.sort()
        print "%f %f" % (cc_trials[i_lo], cc_trials[i_hi])
        #print cc_trials
        cc_intervals[0].append(cc_trials[i_lo])
        cc_intervals[1].append(cc_trials[i_hi])

    ############################################################################
    for i in range(0,3):
        subplots_2d[i+3].scatter(xplt[i],yplt[i],s=2)

        subplots_vars[i+3].scatter(ysub[i],xsub[i],s=2)

    ############################################################################
    print "----------------------"
    for i in range(0,3):
        output = "Mean/sigma: %8.4f %8.4f" % (mean[i], sigma[i])
        print output
        outfile.write(output)
    print "---"
    text_out = ""
    for i in range(0,len(cc)):

        cc_int_lo = cc_intervals[0][i]
        cc_int_hi = cc_intervals[1][i]

        output = " -----\n"
        output += "cc from sample:            %8.5f\n" % (cc[i][0])
        output += "Bootstrap samples mean:    %8.5f\n" % (cc[i][1])
        output += "Bias from bootstrap:       %8.5f\n" % (cc[i][0]-cc[i][1])

        output += "Bootstrap conf. interval:  %8.5f - %8.5f" % (cc_int_lo, cc_int_hi)
        if cc_int_lo>0.0 or cc_int_hi<0.0:
            output += " --- THIS IS INCONSISTENT WITH 0!!!!!!!\n"
            if i>=3:
                text_out += "\\textcolor{red}{\\bf(%5.2f,%5.2f)} " % (cc_int_lo, cc_int_hi)
        else:
            output += "\n"
            if i>=3:
                text_out += "(%5.2f,%5.2f) " % (cc_int_lo, cc_int_hi)

        print output
        outfile.write(output)

        if i==3 or i==4:
            text_out += " & "

        if i==5:
            text_out += " \\\\\n"
            text_out_file.write(text_out)

    ################################################################################
    # the histogram of the data
    nplots = len(cc_all_trials)
    for i in range(0,nplots):
        npts = len(cc_all_trials[i])
        xtemp_interval = []
        for j in range(0,npts):
            if j>=i_lo and j<=i_hi:
                xtemp_interval.append(cc_all_trials[i][j])
        my_range = 1.4*abs(cc_intervals[1][i]-cc_intervals[0][i])
        #print my_range
        range_lo = cc[i][0] - (my_range/2.0)
        range_hi = cc[i][0] + (my_range/2.0)

        # Set the range to include 0 if it does not already
        if range_lo>0.0:
            range_lo=-0.01
        if range_hi<0.0:
            range_hi=0.01

        n, bins, patches = subplots[i].hist(cc_all_trials[i], bins=50, range=(range_lo,range_hi),\
                normed=0, facecolor='green', alpha=0.99, histtype='stepfilled')
        n, bins, patches = subplots[i].hist(xtemp_interval, bins=50, range=(range_lo,range_hi), \
                normed=0, facecolor='red', alpha=0.99, histtype='stepfilled')

        subplots[i].set_xlabel('Corr. coeff trials')
        subplots[i].set_ylabel('# events')
        subplots[i].get_xaxis().set_major_locator(MaxNLocator(4))
        subplots[i].axvline(x=cc[i][0], color='black',linewidth=4)
        subplots[i].axvline(x=0.0,color='blue',linestyle=':',linewidth=2)

    ############################################################################
    ############################################################################
    # Format the 2d plots
    for i in range(0,6):

        title = ""
        if i==0:
            title = r"$\hat{ \rho }^* (m_{ES},\Delta E)$"
        elif i==1:
            title = r"$\hat{ \rho }^* (m_{ES},NN)$"
        elif i==2:
            title = r"$\hat{ \rho }^* (\Delta E,NN)$"
        elif i==3:
            title = r"$\hat{ \rho }^* (m_{ES},NN)$"
        elif i==4:
            title = r"$\hat{ \rho }^* (\Delta E,NN)$"
        elif i==5:
            title = r"$\hat{ \rho }^* (\Delta E,NN)$"

        subplots[i].text(0.6,0.99, title, transform=subplots[i].transAxes, \
                bbox=dict(alpha=1.0,facecolor='white'),fontsize=16)

        subplots_2d[i].set_xlim(xlo[0],xhi[0])
        subplots_2d[i].set_ylim(xlo[1],xhi[1])
        subplots_2d[i].set_xlabel('$m_{ES}$')
        subplots_2d[i].set_ylabel('$\Delta E$')

        subplots_nn[i].set_xlabel('NN output')
        subplots_nn[i].set_ylabel('\# events')

        labels = ['$m_{ES}$','$\Delta E$','NN output']

        xaxis = 0
        yaxis = 0
        if i==0:
            xaxis = 0
            yaxis = 1
        elif i==1:
            xaxis = 0
            yaxis = 2
        elif i==2:
            xaxis = 1
            yaxis = 2
        elif i==3:
            xaxis = 2
            yaxis = 0
        elif i==4:
            xaxis = 2
            yaxis = 1
        elif i==5:
            xaxis = 2
            yaxis = 1


        subplots_vars[i].set_xlim(xlo[xaxis],xhi[xaxis])
        subplots_vars[i].set_ylim(xlo[yaxis],xhi[yaxis])
        subplots_vars[i].set_xlabel(labels[xaxis])
        subplots_vars[i].set_ylabel(labels[yaxis])
        subplots_vars[i].get_xaxis().set_major_locator(MaxNLocator(4))
        subplots_vars[i].get_yaxis().set_major_locator(MaxNLocator(6))

    # Adjust the spacing on the plots.
    for i in range(0,8):
        figs[i].subplots_adjust(left=0.1, wspace=0.5, hspace=0.5, bottom=0.2)
        name = "matplotlib_plots/%s_%i.pdf" % (tag,i)
        print "Saving file %s" % (name)
        figs[i].savefig(name)

    ############################################################################
    if options.batch:
        exit(1)
    else:
        plt.show()

    exit(1)


################################################
if __name__ == "__main__":
    main(sys.argv)

