import matplotlib.pylab as plt
import numpy as np

import lichen.lichen as lch

import sys

tag = ""
if len(sys.argv)>1:
    tag = sys.argv[1]

# Small number of events. 
# Produces different numbers for cuts
#nsig = 50
#nbkg = 200
#nbinsx = 100
#nbinsy = 100

nsig = 100000
nbkg = 20000
nbinsx = 100
nbinsy = 100

################################################################################
# Generate the data
################################################################################
sig_x = np.random.normal(0.895,0.010,nsig)
bkg_x = 0.2*np.random.random(nbkg) + 0.8
tot_x = sig_x.copy()
tot_x = np.append(tot_x,bkg_x.copy())

sig_y = np.random.exponential(0.6,nsig)
bkg_y = 3.0*np.random.random(nbkg) 
tot_y = sig_y.copy()
tot_y = np.append(tot_y,bkg_y.copy())

################################################################################
# Plot it. 
################################################################################

################################################################################
# Plot the signal
################################################################################
plt.figure()
lch.hist_err(sig_x,bins=nbinsx,range=(0.8,1.0),linewidth=2)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
name = "Plots/optstudy_sigx0%s.png" % (tag); plt.savefig(name)

plt.figure()
lch.hist_err(sig_y,bins=nbinsy,range=(0,3.0),linewidth=2)
plt.xlim(0,3.0)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'$\tau$ (arbitrary)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
name = "Plots/optstudy_sigy0%s.png" % (tag); plt.savefig(name)

################################################################################
# Plot the background
################################################################################
plt.figure()
lch.hist_err(bkg_x,bins=nbinsx,range=(0.8,1.0),linewidth=2)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
name = "Plots/optstudy_bkgx0%s.png" % (tag); plt.savefig(name)

plt.figure()
lch.hist_err(bkg_y,bins=nbinsy,range=(0,3.0),linewidth=2)
plt.xlim(0,3.0)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'$\tau$ (arbitrary)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
name = "Plots/optstudy_bkgy0%s.png" % (tag); plt.savefig(name)

################################################################################
# Plot the combined signal and background
################################################################################
plt.figure()
lch.hist_err(tot_x,bins=nbinsx,range=(0.8,1.0),linewidth=2)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
ymax = plt.ylim()[1]
# Draw two lines around mass peak
plt.plot([0.87,0.87],[0,ymax],'r-')
plt.plot([0.92,0.92],[0,ymax],'r-')
plt.ylim(0,ymax)
name = "Plots/optstudy_bothx0%s.png" % (tag); plt.savefig(name)

plt.figure()
lch.hist_err(tot_y,bins=nbinsy,range=(0,3.0),linewidth=2)
plt.xlim(0,3.0)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'$\tau$ (arbitrary)',fontsize=24)
plt.ylabel(r'# events',fontsize=24)
# Draw the line and arrow
ymax = plt.ylim()[1]
plt.plot([2.0,2.0],[0,ymax],'r-')
plt.arrow(2.0-0.05,ymax/2.0,-0.5,0,lw=3,head_width=ymax/20,head_length=0.05,fc='k',ec='k')
name = "Plots/optstudy_bothy0%s.png" % (tag); plt.savefig(name)

################################################################################
# Make some cuts on tau
################################################################################
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

################################################################################
# Make the plots of the efficiency
################################################################################
plt.figure()
plt.plot(nb_pts,ns_pts,'ko')
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)
plt.xlabel(r'Background rejection (fractional)',fontsize=24)
plt.ylabel(r'Signal efficiency (fractional)',fontsize=24)
name = "Plots/optstudy_sigbkgcurve%s.png" % (tag); plt.savefig(name)

################################################################################
# Make the plots of the FOM
################################################################################
plt.figure()
plt.plot(mxpts,mpts0,'ko')
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.20)
plt.xlabel(r'$\tau$ value',fontsize=24)
plt.ylabel(r'$S/\sqrt{S+B}$',fontsize=36)
name = "Plots/optstudy_fom0%s.png" % (tag); plt.savefig(name)

plt.figure()
plt.plot(mxpts,mpts1,'bo')
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.20)
plt.xlabel(r'$\tau$ value',fontsize=24)
plt.ylabel(r'$S/\sqrt{B}$',fontsize=36)
name = "Plots/optstudy_fom1%s.png" % (tag); plt.savefig(name)

plt.figure()
plt.plot(mxpts,mpts2,'ro')
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.20)
plt.xlabel(r'$\tau$ value',fontsize=24)
plt.ylabel(r'$\epsilon/(\sqrt{B}+\frac{a}{2}$)',fontsize=36)
name = "Plots/optstudy_fom2%s.png" % (tag); plt.savefig(name)

index = mpts0.index(max(mpts0))
mxpt = mxpts[index]
maxpts = max(mpts0)
nspt = ns_pts[index]*nsig
nbpt = nb_pts[index]*nbkg
print "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (maxpts,mxpt,nspt,nbpt,nspt/np.sqrt(nspt+nbpt),nspt/np.sqrt(nbpt))

index = mpts1.index(max(mpts1))
mxpt = mxpts[index]
maxpts = max(mpts1)
nspt = ns_pts[index]*nsig
nbpt = nb_pts[index]*nbkg
print "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (maxpts,mxpt,nspt,nbpt,nspt/np.sqrt(nspt+nbpt),nspt/np.sqrt(nbpt))

index = mpts2.index(max(mpts2))
mxpt = mxpts[index]
maxpts = max(mpts2)
nspt = ns_pts[index]*nsig
nbpt = nb_pts[index]*nbkg
print "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (maxpts,mxpt,nspt,nbpt,nspt/np.sqrt(nspt+nbpt),nspt/np.sqrt(nbpt))


#plt.show()
