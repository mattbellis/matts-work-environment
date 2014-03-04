#!/usr/bin/env python

import sys, numpy

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
    nblock = int(sys.argv[3])
    
    arr=numpy.zeros([dim,dim])


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

    for ib in range(nblock):        
        output = ""
        for jb in range(nblock):
            size = int(dim/nblock)
            ib_start = ib*size
            jb_start = jb*size
            ib_stop = (ib+1)*size
            jb_stop = (jb+1)*size
            #if ib != jb and jb > ib:
            #    ib_start = ib_start +1
            #    jb_stop = jb_stop -1
            
            for i in range(ib_start,ib_stop):
                for j in range(jb*size,jb_stop):
                    locali = i-ib*size
                    localj = j-jb*size
                    
                    entry = 0
                    fgcolor = 37
                    
                    # These are the calculations that will be made
                    if i!=j and locali>=localj and ib <=jb:
                        
                        
                        entry = 1
                        arr[i,j]= arr[i,j]+1
                        fgcolor = 36

                        # Arbitrarily change the color at some point. Eventually,
                        # this point would be replaced by a smarter algorithm that maps
                        # onto which thread calculates which entries.
                        if j%2==1:
                            fgcolor = 33

                            # These are empty entries that we don't want to waste
                            # time calculating.

                    elif  i!=j and locali>localj and ib > jb:
                        entry = 1
                        arr[i,j]= arr[i,j]+1
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

                    output += "\033[40m\033[%dm%d \033[0m" % (fgcolor,entry)

        #print output

    print '-----------'
    print arr

    output=""
    flag=0
    for i in range(0,dim):
        for j in range(0,i):
            if arr[i,j]+arr[j,i] != 1:
                print 'problem with ',i,j, ' = ', arr[i,j]+arr[j,i] 
                flag=flag + 1

    
    output=""
    for i in range(0,dim):
        for j in range(0,dim):
            if arr[i,j]==1:
                if (((i - (i%size))/size)%2) == (((j - (j%size))/size)%2) :
                    entry=1
                    if j%2:
                        fgcolor=31
                    else:
                        fgcolor=35
                else :
                    entry=1
                    if j%2:
                        fgcolor=34
                    else:
                        fgcolor=33 
            else :
                entry = arr[i,j]
                fgcolor = 39
            output += "\033[40m\033[%dm%d \033[0m" % (fgcolor,entry)
        output+="\n"


    print output
    
    if flag:
        print "Incomplete coverage: missing",flag, "pairs"
    else :
        print "coverage complete"
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

