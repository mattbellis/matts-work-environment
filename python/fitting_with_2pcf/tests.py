import fit2pcf_tools as ftools
import numpy as np
import matplotlib.pylab as plt

ndim = 2
npts = 1000

# x = data[0]
# y = data[1]
# etc...

#data0,data1 = ftools.gen_flat(ndim,npts)

################################################################################
# Gen data and MC
################################################################################
# Data
data0 = ftools.dbl_gaussian_2D(width=0.3,npts=200)
data1 = 6*np.random.random((ndim,400)) + 2

sigd = np.hstack([data0,data1])
sigr = ftools.gen_randoms(sigd,npts)


# MC signal
MCsigd = ftools.dbl_gaussian_2D(width=0.3, npts=10*npts)
MCsigr = ftools.gen_randoms(MCsigd,10*npts)
# MC background 
MCbkgd = 6*np.random.random((ndim,10*npts)) + 2
MCbkgr = ftools.gen_randoms(MCbkgd,10*npts)
################################################################################
################################################################################
# Mix cocktail
################################################################################
MCd = ftools.mix_cocktail(MCsigd,MCbkgd,600,800)
MCr = ftools.gen_randoms(MCd,npts)

print len(MCd[0])
print len(MCr[0])

#data0,data1 = bkg0,bkg1
data0,data1 = sigd,sigr
w,dd,rr,dr,xbins,werr = ftools.twopcf(data0,data1,nbins=100)
print werr
print w*werr

plt.figure()
plt.subplot(2,2,1)
plt.plot(data0[0],data0[1],'.',markersize=2)
plt.subplot(2,2,2)
plt.plot(data1[0],data1[1],'.',markersize=2)

plt.subplot(2,1,2)
#plt.plot(xbins,w,'o')
plt.errorbar(xbins,w,yerr=w*werr,fmt='o')
plt.xlim(0,8)
plt.ylim(-1,4)

# Do the MC cocktail
data0,data1 = MCd,MCr
w,dd,rr,dr,xbins,werr = ftools.twopcf(data0,data1,nbins=100)
print werr
print w*werr

plt.figure()
plt.subplot(2,2,1)
plt.plot(data0[0],data0[1],'.',markersize=2)
plt.subplot(2,2,2)
plt.plot(data1[0],data1[1],'.',markersize=2)

plt.subplot(2,1,2)
#plt.plot(xbins,w,'o')
plt.errorbar(xbins,w,yerr=w*werr,fmt='o')
plt.xlim(0,8)
plt.ylim(-1,4)



plt.show()

