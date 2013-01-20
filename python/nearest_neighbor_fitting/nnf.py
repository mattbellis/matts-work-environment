import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats

nsig = 200
sig = stats.norm.rvs(loc=5,scale=1,size=nsig)


nbkg = 1000
bkg = np.array([])
while len(bkg)<nbkg:
    temp = stats.expon.rvs(loc=0,scale=6,size=nbkg)
    index0 = temp>0
    index1 = temp<10
    index = index0*index1
    remainder = nbkg-len(bkg)
    bkg = np.append(bkg,temp[index][0:remainder])

print len(bkg)

data = sig.copy()
data = np.append(data,bkg)


plt.figure()
plt.hist(data,bins=50)

template0 = stats.norm.rvs(loc=5,scale=1,size=500)

nt1 = 2000
template1 = np.array([])
while len(template1)<nt1:
    temp = stats.expon.rvs(loc=0,scale=6,size=nt1)
    index0 = temp>0
    index1 = temp<10
    index = index0*index1
    remainder = nt1-len(template1)
    template1 = np.append(template1,temp[index][0:remainder])


plt.figure()
plt.hist(template0,bins=50,color='r')
plt.figure()
plt.hist(template1,bins=50,color='r')

distances_t0 = np.zeros(len(data))
distances_t1 = np.zeros(len(data))
for i,x in enumerate(data):
    dist = np.abs(template0-x)
    temp = np.sort(dist)
    distances_t0[i] = 1.0/temp[20]
    print temp[30]

    dist = np.abs(template1-x)
    temp = np.sort(dist)
    distances_t1[i] = 1.0/temp[20]
    print temp[30]

    #print "---------"
    #print temp[0:10]
    #print temp[-10:-1]

print distances_t0
print distances_t1

plt.figure()
plt.hist(distances_t0,bins=50,color='g')
plt.figure()
plt.hist(distances_t1,bins=50,color='g')


plt.show()

