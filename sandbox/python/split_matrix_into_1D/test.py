#!/usr/bin/env python

import sys

dim = int(sys.argv[1])
nthreads = int(sys.argv[2])

nentries = dim*dim - dim

print nentries 

ncalcs_per_thread = nentries/nthreads

print ncalcs_per_thread

print '-----------'
for i in range(0,dim):
    output = ""
    for j in range(0,dim):
        if i!=j and i>j:
            output += "1 "
        else:
            output += "0 "

    print output

print '-----------'
sum = 0
for i in range(0,dim):
    ncalcs_on_a_column = dim - i - 1

    sum += ncalcs_on_a_column 

    print "%d %d %d %d %d" % (i, ncalcs_on_a_column, sum, sum/ncalcs_per_thread, sum%ncalcs_per_thread)



