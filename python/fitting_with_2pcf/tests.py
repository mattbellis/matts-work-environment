################################################################################
import fit2pcf_tools as ftools
import numpy as np
import matplotlib.pylab as plt

import lichen.iminuit_fitting_utilities as fitutils

import iminuit as minuit

################################################################################

ndim = 2
npts = 1000

nbins = 100

ndata = 0

# x = data[0]
# y = data[1]
# etc...

#data0,data1 = ftools.gen_flat(ndim,npts)

################################################################################
# Gen data and MC
################################################################################
# Data
data0 = ftools.dbl_gaussian_2D(width=0.3,npts=200)
data1 = 6*np.random.random((ndim,800)) + 2

sigd = np.hstack([data0,data1])
sigr = ftools.gen_randoms(sigd,10*npts)

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
#data0,data1 = bkg0,bkg1
data0,data1 = sigd,sigr
w,dd,rr,dr,xbins,werr = ftools.twopcf(data0,data1,nbins=50)
#print werr
#print w*werr
wdata,wdataerr = w,werr

plt.figure()
plt.subplot(2,2,1)
plt.plot(data0[0],data0[1],'.',markersize=2)
plt.subplot(2,2,2)
plt.plot(data1[0],data1[1],'.',markersize=2)

plt.subplot(2,1,2)
#plt.plot(xbins,w,'o')
plt.errorbar(xbins,w,yerr=werr,fmt='o')
plt.xlim(0,8)
plt.ylim(-1,4)


################################################################################
################################################################################

################################################################################
# Chi square function for minuit.
################################################################################
def chisq_minuit(data,p,parnames,params_dict):

    wdata = data[0]
    wdataerr = data[1]
    MCsig = data[2]
    MCbkg = data[3]

    nsig = p[parnames.index('nsig')]
    #nbkg = p[parnames.index('nbkg')]
    nbkg = ndata-nsig

    MCd_subset = ftools.mix_cocktail(MCsig,MCbkg,int(nsig),int(nbkg))
    MCr_subset = ftools.gen_randoms(MCd_subset,5*int(nsig+nbkg))

    w,dd,rr,dr,xbins,werr = ftools.twopcf(MCd_subset,MCr_subset,nbins=50)

    chi2,ndof = ftools.chisq_compare(wdata,w,wdataerr,werr)
    print "chi2,ndof,nsig,nbkg: ",chi2,ndof,nsig,nbkg
    #exit()

    return chi2
################################################################################
    
ndata = 5*float(len(sigd[0]))

params_dict = {}
params_dict['nsig'] = {'fix':False,'start_val':0.4*ndata,'limits':(0.0,ndata),'error':100.0}
params_dict['nbkg'] = {'fix':True,'start_val':0.6*ndata,'limits':(0.0,ndata),'error':1.0}

params_names,kwd = fitutils.dict2kwd(params_dict,verbose=True)

# For chi-squared method
kwd['errordef'] = 1.0
kwd['print_level'] = 2

data = [wdata,wdataerr,MCsigd,MCbkgd]

print "HERE"
print kwd

f = fitutils.Minuit_FCN([data],params_dict,chisq_minuit)

m = minuit.Minuit(f,**kwd)

print "THERE"
m.print_param()


'''
m.migrad()
##m.hesse()
#m.minos()

#print m.get_fmin()
#exit()

values = m.values
errors = m.errors
#errors = m.get_merrors()

print "Values:"
print values
print "Errors:"
print errors


################################################################################
MCd = ftools.mix_cocktail(MCsigd,MCbkgd,values["nsig"],ndata-values["nsig"])
MCr = ftools.gen_randoms(MCd,5*npts)

print len(MCd[0])
print len(MCr[0])

# Do the MC cocktail
data0,data1 = MCd,MCr
w,dd,rr,dr,xbins,werr = ftools.twopcf(data0,data1,nbins=50)
#print werr
#print w*werr
wMC,wMCerr = w,werr

plt.figure()
plt.subplot(2,2,1)
plt.plot(data0[0],data0[1],'.',markersize=2)
plt.subplot(2,2,2)
plt.plot(data1[0],data1[1],'.',markersize=2)

plt.subplot(2,1,2)
#plt.plot(xbins,w,'o')
plt.errorbar(xbins,w,yerr=werr,fmt='o')
plt.xlim(0,8)
plt.ylim(-1,4)
'''


##################### RUN SOME TESTS ##########################################
print "TESTING........>"
for i in range(400,600,10):
    MCd = ftools.mix_cocktail(MCsigd,MCbkgd,i,ndata-i)
    MCr = ftools.gen_randoms(MCd,5*npts)

    # Do the MC cocktail
    data0,data1 = MCd,MCr
    w,dd,rr,dr,xbins,werr = ftools.twopcf(data0,data1,nbins=50)
    chi2,ndof = ftools.chisq_compare(wdata,w,wdataerr,werr)
    print i,chi2

plt.show()

