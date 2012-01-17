#!/usr/bin/env python

import sys

################################################################################
################################################################################
def main():

    # Read in the command line entries
    # Check that the usage is correct
    if len(sys.argv) < 3:
        print "\nUsage:\n\t%s \033[40m\033[31m%s\033[0m \033[40m\033[31m%s\033[0m\n" % \
                (sys.argv[0],'<num galaxies>', '<num total threads>')
        exit(-1)

    dim = int(sys.argv[1])
    nthreads = int(sys.argv[2])

    nentries = dim*dim - dim

    ncalcs_per_thread = nentries/nthreads

    print "Original matrix: %dx%d" % (dim,dim)
    print "Unique entries: %d" % (nentries)
    print "Number of threads: %d" % (nthreads)
    print "Number of calculations per thread: %d" % (ncalcs_per_thread)
    print "Left over calculations: %d" % (nentries%nthreads)

    ############################################################################
    

    print '-----------'
    print '  Matrix   '
    print '-----------'
    for i in range(0,dim):
        output = ""
        for j in range(0,dim):
            entry = 0
            fgcolor = 37

            # These are the calculations that will be made
            if i!=j and i>j:
                entry = 1
                fgcolor = 36

                # Arbitrarily change the color at some point. Eventually,
                # this point would be replaced by a smarter algorithm that maps
                # onto which thread calculates which entries.
                if j%2==1:
                    fgcolor = 33

            # These are empty entries that we don't want to waste
            # time calculating.
            else:
                entry = 0
                fgcolor = 39

            output += "\033[40m\033[%dm%5.2f \033[0m" % (fgcolor,i)

        print output

    print '-----------'



    ############################################################################
    # Trying to figure out how to break this up evenly
    ############################################################################
    '''
    print "----------------------------- "
    print "   scratch diagnostic output "
    print "----------------------------- "
    print "----------------------------- "
    print "----------------------------- "
    print "----------------------------- "
    sum = 0
    for i in range(0,dim):
        ncalcs_on_a_column = dim - i - 1

        sum += ncalcs_on_a_column 

        print "%d %d %d %d %d" % (i, ncalcs_on_a_column, sum, sum/ncalcs_per_thread, sum%ncalcs_per_thread)
    '''



################################################################################
################################################################################
if __name__ == '__main__':
      main()

