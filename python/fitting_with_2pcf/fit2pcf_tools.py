from scipy.spatial.distance import cdist,pdist
import numpy as np

################################################################################
def pair_counts(data0,data1,same=False):

    combined = data0.transpose()
    combined1 = data1.transpose()

    dist = None

    if same:
        dist = pdist(combined,'euclidean')
    else:
        dist = cdist(combined,combined1,'euclidean')

    return dist.flatten()
################################################################################

################################################################################
def twopcf(data,random,nbins=100,cfrange=None):

    nd = len(data[0])
    nr = len(random[0])

    dd_dists = pair_counts(data,data,same=True)
    rr_dists = pair_counts(random,random,same=True)
    dr_dists = pair_counts(data,random)

    if cfrange==None:
        minrange = min(dr_dists)
        maxrange = max(dr_dists)
        cfrange = (minrange,maxrange)

    dd = hist=np.histogram(dd_dists,bins=nbins,range=cfrange)
    rr = hist=np.histogram(rr_dists,bins=nbins,range=cfrange)
    dr = hist=np.histogram(dr_dists,bins=nbins,range=cfrange)

    # Pull out the bin edges (nx + 1)
    xbins = dd[1]
    xbinwidth = xbins[1]-xbins[0]
    xbins = xbins[0:-1] + xbinwidth/2.0
    #print "xbins"
    #print dd
    #print xbins
    #print len(xbins)

    dd = dd[0].astype(float)
    rr = rr[0].astype(float)
    dr = dr[0].astype(float)

    # Do some estimate of the fractional uncertainty
    #werr = np.sqrt((np.sqrt(dd)/dd)**2 + (np.sqrt(rr)/rr)**2 + (np.sqrt(dr)/dr)**2)
    werr = np.sqrt(dd)/dd + np.sqrt(rr)/rr + np.sqrt(dr)/dr

    #print type(dd)
    #print dd

    norm_dd = 0.5*(nd*nd - nd)
    norm_rr = 0.5*(nr*nr - nr)
    norm_dr = nd*nr

    print norm_dd, norm_dr, norm_rr

    dd /= norm_dd
    rr /= norm_rr
    dr /= norm_dr

    w = (dd - 2*dr + rr)/rr


    return w,dd,rr,dr,xbins,werr
################################################################################

################################################################################
def gen_flat(ndim,npts):

    data0 = np.random.random((ndim,npts))
    data1 = np.random.random((ndim,npts))

    return data0,data1
################################################################################

################################################################################
def dbl_gaussian_2D(mean=5,width=1.,npts=1000):

    # Double Gaussian
    datax = np.random.normal(mean,width,npts)
    datay = np.random.normal(mean,width,npts)
    data = np.array([datax,datay])

    return data
################################################################################

################################################################################
def gen_randoms(data,npts):

    random = []
    for d in data:
        r = (max(d)-min(d))*np.random.random(npts) + min(d)
        random.append(r)

    random = np.array(random)

    return random


################################################################################
def mix_cocktail(data0,data1,nd0=10,nd1=20):

    sample0 = data0.copy()
    np.random.shuffle(sample0.transpose())
    sample0 = sample0.transpose()[0:nd0]
    sample0 = sample0.transpose()

    sample1 = data1.copy()
    np.random.shuffle(sample1.transpose())
    sample1 = sample1.transpose()[0:nd1]
    sample1 = sample1.transpose()

    cocktail = np.hstack([sample0,sample1])

    return cocktail



