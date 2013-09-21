#!/usr/bin/env python

import sys

################################################################################
################################################################################
def main():

    # Read in the command line entries
    # Check that the usage is correct
    if len(sys.argv) < 4:
        print "\nUsage:\n\t%s \033[40m\033[31m%s\033[0m \033[40m\033[31m%s\033[0m \033[40m\033[31m%s\033[0m\n" % \
                (sys.argv[0],'<num galaxies>', '<num total threads>', '<which thread to highlight>')
        exit(-1)

    dim = int(sys.argv[1])
    nthreads = int(sys.argv[2])
    # Which thread to highlight
    which_thread = int(sys.argv[3])

    nentries = dim

    ncalcs_per_thread = nentries/nthreads

    print "Length of 1D array: %d" % (dim)
    print "Number of threads: %d" % (nthreads)
    print "Number of calculations per thread: %d" % (ncalcs_per_thread)
    print "Number of leftover calculations: %d" % (nentries%nthreads)

    ############################################################################
    
    # Figure the starting and ending point for that particular thread
    lo = which_thread * ncalcs_per_thread
    hi = lo + ncalcs_per_thread 
    #print lo
    #print hi

    print '-----------'
    print '  Array   '
    print '-----------'
    output = ""
    for i in range(0,nentries):
        entry = 1
        fgcolor = 37

        if i>=lo and i<hi:
            fgcolor = 31

        # These are empty entries that we don't want to waste
        # time calculating.
        else:
            fgcolor = 39

        output += "\033[40m\033[%dm%d \033[0m" % (fgcolor,entry)

    print output

    print '-----------'




################################################################################
################################################################################
if __name__ == '__main__':
      main()

