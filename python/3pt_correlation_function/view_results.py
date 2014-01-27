import numpy as np
import matplotlib.pylab as plt

import sys

ngals = 10
nbins = 16

nd = ngals*1000
nr = ngals*1000

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

i = 3
j = 4
k = 5
print ddd[i][j][k],ddd[i][k][j],ddd[j][i][k],ddd[j][k][i],ddd[k][i][j],ddd[k][j][i]
print ddr[i][j][k],ddr[i][k][j],ddr[j][i][k],ddr[j][k][i],ddr[k][i][j],ddr[k][j][i]
print drr[i][j][k],drr[i][k][j],drr[j][i][k],drr[j][k][i],drr[k][i][j],drr[k][j][i]
print rrr[i][j][k],rrr[i][k][j],rrr[j][i][k],rrr[j][k][i],rrr[k][i][j],rrr[k][j][i]

dddnorm = nd*(nd-1)*(nd-2)/6.0
ddrnorm = nd*(nd-1)*(nr)/2.0
drrnorm = nd*(nr)*(nr-1)/2.0
rrrnorm = nr*(nr-1)*(nr-2)/6.0

print dddnorm
print ddrnorm
print drrnorm
print rrrnorm


tot = 0
for dd in rrr:
    for d in dd:
        tot += sum(d)

print tot

ddd = ddd.astype('float')/dddnorm
ddr = ddr.astype('float')/ddrnorm
drr = drr.astype('float')/drrnorm
rrr = rrr.astype('float')/rrrnorm

#w3 = (ddd - 3*ddr + 3*drr - rrr)/rrr.astype('float')
w3 = (ddd - 3*ddr + 3*drr - rrr)
