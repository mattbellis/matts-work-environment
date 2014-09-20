import numpy as np
import matplotlib.pylab as plt

import sys

from scipy import interpolate


infiles = ['Fe_FNDA1_output.csv','Fe_FNDA2_output.csv']
element = 'Fe'
#infiles = ['Ni_FNDA1_output.csv','Ni_FNDA2_output.csv']
#element = 'Ni'

cis = [None,None]
yis = [None,None]
yerris = [None,None]
c56 = [None,None]
sim_deltas = [None,None]
sim_deltas_fake0 = [None,None]
sim_deltas_fake1 = [None,None]

markers = ['o','s']
for i in range(0,2):
    cis[i],yis[i],yerris[i],c56[i],sim_deltas[i],sim_deltas_fake0[i],sim_deltas_fake1[i]    = np.loadtxt(infiles[i],delimiter=',',dtype=str,usecols=(3,8,9,5,11,12,13),unpack=True)

    cis[i] = cis[i][cis[i]!=''].astype(float)
    yis[i] = yis[i][yis[i]!=''].astype(float)
    yerris[i] = yerris[i][yerris[i]!=''].astype(float)
    c56[i] = c56[i][c56[i]!=''].astype(float)
    sim_deltas[i] = sim_deltas[i][sim_deltas[i]!=''].astype(float)
    sim_deltas_fake0[i] = sim_deltas_fake0[i][sim_deltas_fake0[i]!=''].astype(float)
    sim_deltas_fake1[i] = sim_deltas_fake1[i][sim_deltas_fake1[i]!=''].astype(float)
    #print cis[i]


plt.figure(figsize=(12,6))

label = r"%s, FNDA 1 data" % (element)
plt.errorbar(cis[0],yis[0],yerr=yerris[0],fmt=markers[0],markersize=10,label=label,color='r')
label = r"%s, FNDA 2 data" % (element)
plt.errorbar(cis[1],yis[1],yerr=yerris[1],fmt=markers[1],markersize=10,label=label,color='b')

'''
label = r"%s, FNDA 1 best fit" % (element)
plt.plot(c56[0],sim_deltas[0],'r--',linewidth=1,label=label)
label = r"%s, FNDA 2 best fit" % (element)
plt.plot(c56[1],sim_deltas[1],'b--',linewidth=1,label=label)
'''

print c56[0]
print c56[1]
print len(c56[0])
print len(c56[1])

xsims = np.linspace(0.02,0.99,100)
if element=="Fe":
    xsims = np.linspace(0.011,0.99,100)
    interpolate_sim0 = interpolate.interp1d(c56[0],sim_deltas[0])
    interpolate_sim1 = interpolate.interp1d(c56[1],sim_deltas[1])
elif element=="Ni":
    xsims = np.linspace(0.011,0.98,100)
    #xsims = xsims[::-1]
    print "here"
    interpolate_sim0 = interpolate.interp1d(c56[0][::-1],sim_deltas[0][::-1])
    interpolate_sim1 = interpolate.interp1d(c56[1][::-1],sim_deltas[1][::-1])

print xsims

ysims0 = interpolate_sim0(xsims)
ysims1 = interpolate_sim1(xsims)

print ysims0
print ysims1


label = r"%s, Average of fits to two experiments" % (element)
fit_mean = (ysims0+ysims1)/2.0
fit_stddev = np.zeros(len(fit_mean))
for s in [ysims0,ysims1]:
    fit_stddev += (s-fit_mean)**2
fit_stddev /= len(sim_deltas)
fit_stddev = np.sqrt(fit_stddev) 

plt.plot(xsims,fit_mean,'k-',linewidth=3,label=label)
if element=="Ni":
    plt.fill_between(c56[0],sim_deltas_fake0[0],sim_deltas_fake1[0],alpha=0.5,color='grey',linewidth=1,label=label)
elif element=="Fe":
    #plt.fill_between(c56[0],sim_deltas_fake0[0],sim_deltas_fake1[0],alpha=0.5,color='grey',linewidth=1,label=label)
    plt.fill_between(xsims,fit_mean+2*fit_stddev,fit_mean-2*fit_stddev,alpha=0.5,color='grey',linewidth=1,label=label)
#plt.plot(c56[1],'k--',linewidth=3,label=label)

plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Concentration',fontsize=24)
if element=="Fe":
    plt.legend(loc='lower right',fontsize=24) 
elif element=="Ni":
    plt.legend(loc='upper right',fontsize=24) 

plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
name = "%s_both_experiments_delta_vs_concentration.png" % (element)
plt.savefig(name)

plt.show()

