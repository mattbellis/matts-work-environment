import sys
import numpy as np
import matplotlib.pyplot as plt

import csv

################################################################################
# main
################################################################################
def main():
    
    std0 = 'PAW'
    std1 = 'PNI'
    test = 'P40'

    ############################################################################
    # Open the files and read the data.
    ############################################################################

    infilename = 'Mexico_ICPMS.csv'
    if len(sys.argv)>=2:
        infilename = sys.argv[1]

    ############################################################################
    ############################################################################

    # Parse the entire contents of the file into a dictionary
    infile = csv.reader(open(infilename, 'rb'), delimiter=',')

    water = {}
    sample_types = [] # Types of the samples
    samples = [] # Names of the samples
    isotopes = [] # Names of the isotopes

    i = 0
    for row in infile:
        #print row
        if i==0:
            sample_types = row
        elif i==1:
            samples = row
            for name in row: 
                water[name] = np.array([])
        elif i>1:
            for j,name in enumerate(samples): 
                val = row[j]
                if j>0:
                    water[name] = np.append(water[name],float(val))
                elif j==0:
                    isotopes.append(val)

        i += 1

    for st,s in zip(sample_types,samples):
        print "%-16s %-12s" % (st,s)

    #print samples
    
    ############################################################################

    ################################################################################
    # Make a figure on which to plot.
    ################################################################################
    fig0 = plt.figure(figsize=(9,6),dpi=100,facecolor='w',edgecolor='k')
    ax0 = fig0.add_subplot(1,1,1)
    fig0.subplots_adjust(top=0.95,bottom=0.15,right=0.95)
    ################################################################################
    
    ############################################################################
    # Format the plot.
    ############################################################################
    xtitle = r"(%s-%s)/%s" % (test,std0,std0)
    ytitle = r"(%s-%s)/%s" % (test,std1,std1)
    ax0.set_xlabel(xtitle, fontsize=24, weight='bold')
    ax0.set_ylabel(ytitle, fontsize=24, weight='bold')
    #plt.xticks(fontsize=24,weight='bold')
    #plt.yticks(fontsize=24,weight='bold')

    xpts = (water[test]-water[std0])/water[std0]
    ypts = (water[test]-water[std1])/water[std1]
    ax0.scatter(xpts,ypts,s=30)

    # Draw a line with slope 1 for reference.
    xline = np.linspace(-1000,1000,1000)
    yline = np.linspace(-1000,1000,1000)
    ax0.plot(xline,yline)

    print "################################################################################"
    print "Test sample: %s" % (test)
    print "Standard 0:  %s" % (std0)
    print "Standard 1:  %s" % (std1)
    print "--------------------------------------------------------------------------------"
    print "%-12s %12s %12s %12s %12s %12s" % ("isotope","test","standard 0","standard 1","% diff 0","% diff 1")
    print "--------------------------------------------------------------------------------"
    for i,s0,s1,c,pct0,pct1 in zip(isotopes,water[std0],water[std1],water[test],xpts,ypts):
        print "%-12s %12.6f %12.6f %12.6f %12.6f %12.6f" % (i,c,s0,s1,pct0,pct1)

    #ax0.set_xscale('log')
    #ax0.set_yscale('log')
   
    ax0.set_xlim(min(xpts)-(0.1*max(xpts)),1.1*max(xpts))
    ax0.set_ylim(min(ypts)-(0.1*max(ypts)),1.1*max(ypts))

    plt.show()
    #'''

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()

