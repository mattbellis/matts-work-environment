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


#from bootstrapping import *

################################################################################
################################################################################
def main(argv):

    #### Command line variables ####
    parser = OptionParser()
    parser.add_option("--ntp", dest="ntp", default='ntp1', help='ntp')
    parser.add_option("--baryon", dest="baryon", default='LambdaC', \
            help='baryon')
    parser.add_option("--nn-lo", dest="nn_lo", default='-1.0', \
            help="NN low edge.")
    parser.add_option("--tag", dest="tag", default='default', \
            help="Tag for ouput plots.")
    parser.add_option("--zoom", dest="zoom", action='store_true', \
            default=False, help='Zoom in.')
    parser.add_option("--zoom-more", dest="zoom_more", action='store_true', \
            default=False, help='Zoom in more.')
    parser.add_option("--batch", dest="batch", action='store_true', 
            default=False, help='Run in batch mode')

    (options, args) = parser.parse_args()

    baryon = options.baryon
    ntp = options.ntp

    ################################################################################
    # For Latex stuff
    rc('text', usetex=True)
    rc('font', family='serif')
    ################################################################################

    #text_LambdaC_ntp2_TMVA_qqbarandbbar_all_6vars_sideband_cut0.txt
    bkg_samples = ['qqbar','qqbarandbbar']
    num_tmva_vars = ['4','6']
    #s_and_b = ["signalSP","genericSP"]
    s_and_b = ["signalSP","genericQQbar"]
    #s_and_b = ["signalSP","genericBBar"]
    signal = "signalSP"
    backgrounds = ["genericSP","genericQQbar","genericBBbar","sideband"]

    background_tags = ["Bkg is generics:",r"Bkg is $q\bar{q}$:",r"Bkg is $B\bar{B}$:","Bkg is sideband data"]
    bs_tags = [r"trained on $q\bar{q}$",r"trained on $q\bar{q}$ and $B\bar{B}$"]
    n_tags = [", 4 disc. vars.", ", 6 disc. vars."]


    legend_names = []
    filenames = []
    for b,bkg in enumerate(backgrounds):
        legend_names.append([])
        filenames.append([[],[]])
        for j,bs in enumerate(bkg_samples):
            for k,n in enumerate(num_tmva_vars):
                name = "%s %s %s" % (background_tags[b],bs_tags[j],n_tags[k])
                legend_names[b].append(name)
                for i in range(0,2):
                    if i==0:
                        f = "textFiles/text_%s_%s_TMVA_%s_all_%svars_%s_cut0.txt" % \
                                (baryon,ntp,bs,n,signal)
                    else:
                        f = "textFiles/text_%s_%s_TMVA_%s_all_%svars_%s_cut0.txt" % \
                                (baryon,ntp,bs,n,bkg)
                    filenames[b][i].append(f)

    #print filenames

    colors = ['blue','green','red','orange']
    #markers = ['^','s','o','v']
    #markers = ['+','+','+','+']
    markers = ['o','o','o','o']


    ################################################################################
    # Make a figure on which to plot stuff.
    figs = []
    subplots = []
    fig_names = []
    for b,bkg in enumerate(backgrounds):
        zoom_tag = ""
        if options.zoom and not options.zoom_more:
            zoom_tag = "_zoom"
        elif not options.zoom and options.zoom_more:
            zoom_tag = "_zoom_more"
        name = "my_roc_curves_%s_%s%s_%s_%d" % (baryon, ntp, zoom_tag, options.tag, b)

        figs.append(plt.figure(figsize=(6,6), dpi=100, facecolor='w', edgecolor='k'))
        figs[b].set_label(name)
        #
        # Usage is XYZ: X=how many rows to divide.
        #               Y=how many columns to divide.
        #               Z=which plot to plot based on the first being '1'.
        # So '111' is just one plot on the main figure.
        ################################################################################
        subplots.append([])
        for i in range(0,4):
            division = 111
            subplots[b].append(figs[b].add_subplot(division))

    plots = []
    npts = 100
    lo = float(options.nn_lo)
    hi =  1.0
    step = (hi-lo)/npts
    ############################################################################
    # Open the file
    for b,bkg in enumerate(backgrounds):
        plots.append([])
        for i,f in enumerate(filenames[b][0]):

            if b==3:
                print "-------"

            infiles = [None, None]

            infile_name_sig = f
            infile_name_bkg = filenames[b][1][i]

            #print infile_name_sig
            #print infile_name_bkg

            infiles[0] = open(infile_name_sig)
            infiles[1] = open(infile_name_bkg)
            
            nn_output = [[],[]]
            for j in range(0,2):
                for line in infiles[j]:
                    nn_output[j].append(float(line.split()[2]))

            # 0: sig
            # 1: bkg
            pts = [[],[]]
            nsig = float(len(nn_output[0]))
            nbkg = float(len(nn_output[1]))
            for k in range(0,npts+1):
                pts[0].append(0.0)
                pts[1].append(0.0)
                nn_cut = lo + k*step
                for j in range(0,2):
                    for n in nn_output[j]:
                        if n>nn_cut:
                            pts[j][k] += 1.0

                if nsig!=0:
                    pts[0][k] /= nsig
                else:
                    pts[0][k] = 0.0

                if nbkg!=0:
                    pts[1][k] /= nbkg
                else:
                    pts[1][k] = 0.0

                # Do the bkg rejection
                pts[1][k] = 1.0 - pts[1][k]

                if pts[0][k]>0.88 and pts[0][k]<0.92 and b==3:
                    print "%s %6.3f %6.3f %6.3f %6f" % (infile_name_bkg, pts[0][k], pts[1][k], nbkg*(1-pts[1][k]), nn_cut)

            plots[b].append(subplots[b][i].scatter(pts[0],pts[1],s=20,color=colors[i],alpha=1.0,marker=markers[i]))


    # Get the tag for the output files
    #tag = infilename.split('/')[-1].split('.txt')[0]

    #tag += "_ntrials%d_nn%d" % (ntrials, int(float(options.nn_lo)*100))

    #outfilename = "matplotlib_textFiles/%s.txt" % (tag)
    #outfile = open(outfilename,"w+")

    ############################################################################
    # Get all the values
    ############################################################################
    # Plot the curves
    ############################################################################
    print "----------------------"
    ################################################################################
    legends = []
    for b,bkg in enumerate(backgrounds):
        #print bkg
        nplots = len(subplots[b])
        for i in range(0,nplots):
            subplots[b][i].grid(True)
            subplots[b][i].get_xaxis().set_major_locator(MaxNLocator(6))
            if options.zoom:
                subplots[b][i].set_xlim(0.8,1.0)
                subplots[b][i].set_ylim(0.0,0.8)
            elif options.zoom_more:
                subplots[b][i].set_xlim(0.8,1.0)
                subplots[b][i].set_ylim(0.3,0.8)
            else:
                subplots[b][i].set_ylim(0.0,1.3)
            subplots[b][i].set_ylabel('Background rejection')
            subplots[b][i].set_xlabel('Signal efficiency')

        # Adjust the spacing on the plots.
        figs[b].subplots_adjust(left=0.12, wspace=0.5, hspace=0.5)
        #print plots[b]
        #print legend_names[b]
        #subplots[b][i].legend(plots[b], legend_names[b], loc='lower left',scatterpoints=1)
        figs[b].legend(plots[b], legend_names[b], loc='upper right',scatterpoints=1,markerscale=5.0)


    ############################################################################
    # Save the figures
    for fig in figs:
        name = "matplotlib_plots/%s.pdf" % (fig.get_label())
        print name
        fig.savefig(name)
    '''
    name = "matplotlib_plots/%s_0.pdf" % (tag)
    figs[b].savefig(name)
    name = "matplotlib_plots/%s_1.pdf" % (tag)
    fig2.savefig(name)
    name = "matplotlib_plots/%s_2.pdf" % (tag)
    fig3.savefig(name)
    '''

    if options.batch:
        exit(1)
    else:
        plt.show()

    exit(1)


################################################
if __name__ == "__main__":
    main(sys.argv)

