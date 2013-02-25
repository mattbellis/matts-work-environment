import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats

import lichen.pdfs as pdfs
import lichen.iminuit_fitting_utilities as fitutils
import lichen.plotting_utilities as plotutils

import iminuit as minuit

################################################################################
# CoGeNT fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    tot_pdf = np.zeros(len(data[0]))

    n0 = p[parnames.index('num0')]
    n1 = p[parnames.index('num1')]

    print data[0]
    print data[1]

    tot_pdf = n0*data[0] + n1*data[1]

    return tot_pdf

################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data,p,parnames,params_dict):

    ndata = len(data[0])

    flag = p[parnames.index('flag')]

    num_tot = 0.0
    for name in parnames:
        if 'num' in name:
            num_tot += p[parnames.index(name)]

    tot_pdf = fitfunc(data,p,parnames,params_dict)

    likelihood_func = (-np.log(tot_pdf)).sum()

    print num_tot,ndata

    ret = likelihood_func - fitutils.pois(num_tot,ndata)

    return ret

################################################################################

################################################################################
# Generate signal
################################################################################
nsig = 200
sig = stats.norm.rvs(loc=5,scale=1,size=nsig)


################################################################################
# Generate background
################################################################################
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

################################################################################
# Combine signal and background
################################################################################
data = sig.copy()
data = np.append(data,bkg)

plt.figure()
plt.hist(data,bins=50)

################################################################################
# Generate the templates we will use to fit the data
################################################################################
nt0 = 20000
template0 = stats.norm.rvs(loc=5,scale=1,size=nt0)

nt1 = 20000
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

nt_tot = nt0+nt1


################################################################################
# Calculate the distances from data to the templates.
################################################################################
max_dist = 20
distances_t0 = np.zeros(len(data))
distances_t1 = np.zeros(len(data))
for i,x in enumerate(data):
    dist = np.abs(template0-x)
    temp = np.sort(dist)
    distances_t0[i] = 1.0/temp[max_dist]
    #print temp[30]

    dist = np.abs(template1-x)
    temp = np.sort(dist)
    distances_t1[i] = 1.0/temp[max_dist]
    #print temp[30]

    #print "---------"
    #print temp[0:10]
    #print temp[-10:-1]

print "Distances"
print distances_t0
print distances_t1

plt.figure()
plt.hist(distances_t0,bins=50,color='g')
plt.figure()
plt.hist(distances_t1,bins=50,color='g')

############################################################################
# Declare the fit parameters
############################################################################
ranges = [[0,10]]

params_dict = {}
#params_dict['flag'] = {'fix':True,'start_val':args.fit}
params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_x'] = {'fix':True,'start_val':0,'limits':(ranges[0][0],ranges[0][1])}

ncomponents = [200,1000]
for i,val in enumerate(ncomponents):
    name = "num%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(0.0,6000.0)}

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([[distances_t0,distances_t1]],params_dict,emlf_normalized_minuit)

m = minuit.Minuit(f,**kwd)

# For maximum likelihood method.
m.up = 0.5

# Up the tolerance.
m.tol = 1.0

m.migrad()

values = m.values


plt.show()

