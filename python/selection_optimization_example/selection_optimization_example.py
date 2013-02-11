import matplotlib.pylab as plt
import numpy as np

import lichen.lichen as lch

nsig = 200
nbkg = 200
nbinsx = 100
nbinsy = 100

sig_x = np.random.normal(0.895,0.010,nsig)
bkg_x = 0.2*np.random.random(nbkg) + 0.8
tot_x = sig_x.copy()
tot_x = np.append(tot_x,bkg_x.copy())

sig_y = np.random.exponential(0.6,nsig)
bkg_y = 3.0*np.random.random(nbkg) 
tot_y = sig_y.copy()
tot_y = np.append(tot_y,bkg_y.copy())

#plt.figure()
#lch.hist_err(sig_x,bins=nbinsx,range=(0.8,1.0))

#plt.figure()
#lch.hist_err(bkg_x,bins=nbinsx,range=(0.8,1.0))

plt.figure()
lch.hist_err(tot_x,bins=nbinsx,range=(0.8,1.0))
# Draw two lines around mass peak
#plt.plot([0.87,0.87],[0,2e4],'r-')
#plt.plot([0.92,0.92],[0,2e4],'r-')
#plt.ylim(0,15000)


#plt.figure()
#lch.hist_err(sig_y,bins=nbinsy,range=(0,3.0))
#plt.xlim(0,3.0)
#plt.figure()
#lch.hist_err(bkg_y,bins=nbinsy,range=(0,3.0))
#plt.xlim(0,3.0)
plt.figure()
lch.hist_err(tot_y,bins=nbinsy,range=(0,3.0))
plt.xlim(0,3.0)

ns_pts = []
nb_pts = []

mxpts = []
mpts0 = []
mpts1 = []
mpts2 = []

for i in np.arange(3.0,0.1,-0.05):
    #print i,len(sig_y[sig_y>i])
    index0 = sig_y<i
    index1 = sig_x>0.87
    index2 = sig_x<0.92
    index = index0*index1*index2
    nsig_surv = len(sig_y[index])

    index0 = bkg_y<i
    index1 = bkg_x>0.87
    index2 = bkg_x<0.92
    index = index0*index1*index2
    nbkg_surv = len(bkg_y[index])

    print nbkg_surv,nsig_surv

    ns_pts.append(nsig_surv/float(nsig))
    nb_pts.append(1.0-(nbkg_surv/float(nbkg)))

    mxpts.append(i)
    mpts0.append(nsig_surv/np.sqrt(nsig_surv+nbkg_surv))
    mpts1.append(nsig_surv/np.sqrt(nbkg_surv))
    mpts2.append((nsig_surv/float(nsig))/(np.sqrt(nbkg_surv) + (4.0/2.0)))

print ns_pts
print nb_pts

plt.figure()
plt.plot(nb_pts,ns_pts,'ko')

plt.figure()
plt.plot(mxpts,mpts0,'ko')
plt.plot(mxpts,mpts1,'bo')
plt.plot(mxpts,mpts2,'ro')

print "%8.3f %8.3f %8.3f %8.3f" % (max(mpts0),mxpts[mpts0.index(max(mpts0))],ns_pts[mpts0.index(max(mpts0))]*nsig,nb_pts[mpts0.index(max(mpts0))]*nbkg)
print "%8.3f %8.3f %8.3f %8.3f" % (max(mpts1),mxpts[mpts1.index(max(mpts1))],ns_pts[mpts1.index(max(mpts1))]*nsig,nb_pts[mpts1.index(max(mpts1))]*nbkg)
print "%8.3f %8.3f %8.3f %8.3f" % (max(mpts2),mxpts[mpts2.index(max(mpts2))],ns_pts[mpts2.index(max(mpts2))]*nsig,nb_pts[mpts2.index(max(mpts2))]*nbkg)

plt.show()
