import numpy as np
import matplotlib.pylab as plt

import sys

ngals = 10
nbins = 16

inputfilenames = []

for subset in ['DDD','DDR','DRR','RRR']:
    name = "output_files/%s_evenbinning_GPU_naive_%dbin_%dk.dat" % (subset,nbins,ngals)
    inputfilenames.append(name)

ddd = np.zeros((nbins,nbins,nbins),dtype=np.int)
ddr = np.zeros((nbins,nbins,nbins),dtype=np.int)
drr = np.zeros((nbins,nbins,nbins),dtype=np.int)
rrr = np.zeros((nbins,nbins,nbins),dtype=np.int)

for n,filename in enumerate(inputfilenames):

    infile = open(filename,'r')

    lcount = 0
    i = 0
    j = 0
    k = 0
    for line in infile:
        x = np.array(line.split()).astype('int')

        if len(x)==1:
            i = x[0]
            j = 0

        if len(x)==nbins:
            if n==0:
                ddd[i][j] = x
            elif n==1:
                ddr[i][j] = x
            elif n==2:
                drr[i][j] = x
            elif n==3:
                rrr[i][j] = x

            j += 1

        lcount += 1

