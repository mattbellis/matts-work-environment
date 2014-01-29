import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

ngals = 10
nbins = 16

nd = ngals*1000
nr = ngals*1000

inputfilenames = []

for subset in ['DDD','DDR','DRR','RRR']:
    name = "output_files/%s_evenbinning_GPU_naive_%dbin_cartesian_%dk.dat" % (subset,nbins,ngals)
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

binwidth = 48.0/16
x = []
y = []
z = []
sizes = []
for i in range(0,16):
    for j in range(0,16):
        for k in range(0,16):
            if w3[i][j][k]!=0:
                x.append(i*binwidth)
                y.append(j*binwidth)
                z.append(k*binwidth)
                #print w3[i][j][k]
                #print np.log10(w3[i][j][k])+20
                sizes.append(3*(np.log10(w3[i][j][k])+20))
                #sizes.append(10000*w3[i][j][k])

fig = plt.figure(figsize=(7,5),dpi=100)
ax = fig.add_subplot(1,1,1)
ax = fig.gca(projection='3d')
plt.subplots_adjust(top=0.98,bottom=0.02,right=0.98,left=0.02)
#ax.scatter([1,2],[1,2],[1,2],s=[200,100],color='b')
ax.scatter(x,y,z,s=sizes,c=sizes,cmap=plt.cm.hsv)
ax.set_xlabel('Side A of triangle [Mpc]')
ax.set_ylabel('Side B of triangle [Mpc]')
ax.set_zlabel('Side C of triangle [Mpc]')

ax.set_xlim(0,45)
ax.set_ylim(0,45)
ax.set_zlim(0,45)

plt.savefig('3pc_histo.png')

plt.show()
