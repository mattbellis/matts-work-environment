import numpy as np
import matplotlib.pylab as plt
import lichen.lichen as lch
import lichen.pdfs as pdfs
import lichen.iminuit_fitting_utilities as fitutils
import lichen.plotting_utilities as plotutils

import scipy.stats as stats

from datetime import datetime,timedelta

import iminuit as minuit

samples = ['ttbar','t','tbar','wjets','qcd','mu']
samples_label = ['t#bar{t}','Single-Top','#bar{t}',"W#rightarrow#mu#nu",'QCD','data']
#fcolor = [ROOT.kRed+1, ROOT.kMagenta, ROOT.kMagenta, ROOT.kGreen-3,ROOT.kYellow,ROOT.kWhite]
fcolor = ['r', 'm', 'm', 'g','y']
first_guess = [0.0,0.0,0.0,0.0,0.0]
first_guess[0] = 16165.289 # ttbar, what's left over
first_guess[1] = 635.094 # From the COUNTING group! single t
first_guess[2] = 361.837 # From the COUNTING group! single tbar
first_guess[3] = 2240.79 # From the COUNTING group! wjets
first_guess[4] = 4235.99 # Derived from data, QCD stuff


njets_min = 4
njets_max = 4


################################################################################
# Fit function.
################################################################################
def fitfunc(nums,templates):

    tot_pdf = np.zeros(len(templates[0][1]))
    #print nums
    #print templates
    #norm = sum(nums)
    #print "IN FITFUNC CALL:"
    for n,t in zip(nums,templates):
        #n /= norm
        #print n
        #print t[1]
        tot_pdf += n*t[1] # Use the y-values
        #print tot_pdf
    #tot_pdf /= ntot # DO I NEED THIS?
    print nums

    return tot_pdf







################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data_and_pdfs,p,parnames,params_dict):

    data = data_and_pdfs[0]
    templates = data_and_pdfs[1]

    flag = p[parnames.index('flag')]

    num_tot = 0.0
    for name in parnames:
        if 'num' in name:
            num_tot += p[parnames.index(name)]

    tot_pdf = 0.0
    likelihood_func = 0.0
    for j in range(njets_min,njets_max+1):
        nums = []
        for i,s in enumerate(samples[0:-1]):
            name = "num_%s_njets%d" % (s,j)
            nums.append(p[parnames.index(name)])
        tot_pdf = fitfunc(nums,templates[j-njets_min])
        y = data[j-njets_min][1]

        likelihood_func += (-y[y>0]*np.log(tot_pdf[y>0])).sum()

    print "num_tot: ",num_tot
    #ret = likelihood_func - fitutils.pois(num_tot,ndata)
    #num_tot = num00 + num10
    ret = likelihood_func + num_tot
    print "ret: ",ret,likelihood_func,num_tot

    return ret

################################################################################




################################################################################
# Generate the fitting templates
################################################################################

#nbins = 100
ranges = [0.0, 5.0]

################################################################################
# Read in the data.
################################################################################
data = []
for j in range(njets_min,njets_max+1):
    infilename = "templates/output_mu_njets%d.dat" % (j)
    infile = open(infilename,'rb')

    content = np.array(infile.read().split()).astype('float')
    index = np.arange(0,len(content),2)
    x = content[index]
    y = content[index+1]

    # Rebin
    print "nbins: ",len(x)
    index = np.arange(0,len(x),2)
    tempx = (x[index]+x[index+1])/2.0
    tempy = (y[index]+y[index+1])
    x = tempx
    y = tempy

    data.append([x.copy(),y.copy()])
    #plt.figure()
    #plt.plot(x,y,'bo',ls='steps')

################################################################################
# Read in the templates.
################################################################################
templates = []
for j in range(njets_min,njets_max+1):
    templates.append([])
    for i,s in enumerate(samples[0:-1]):
        infilename = "templates/output_%s_njets%d.dat" % (s,j)
        infile = open(infilename,'rb')

        content = np.array(infile.read().split()).astype('float')
        index = np.arange(0,len(content),2)
        x = content[index]
        y = content[index+1]

        # Rebin
        print "nbins: ",len(x)
        index = np.arange(0,len(x),2)
        tempx = (x[index]+x[index+1])/2.0
        tempy = (y[index]+y[index+1])
        x = tempx
        y = tempy

        norm = float(sum(y))
        templates[j-njets_min].append([x.copy(),y.copy()/norm])
        #plt.figure()
        #plt.plot(x,y,'ro',ls='steps')

#plt.show()
#exit()

ndata = []
for d in data:
    ndata.append(float(sum(d[1])))

print "ndata: "
print ndata
#exit()
############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}
params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_x'] = {'fix':True,'start_val':0,'limits':(ranges[0],ranges[1])}
params_dict['ntemplates'] = {'fix':True,'start_val':5,'limits':(1,10)}

for i,s in enumerate(samples[0:-1]):
    for j in range(njets_min,njets_max+1):
        name = "num_%s_njets%d" % (s,j)
        print name
        nd = ndata[j-njets_min]
        if s=='ttbar':
            params_dict[name] = {'fix':False,'start_val':first_guess[i],'limits':(0,nd)}
        else:
            params_dict[name] = {'fix':False,'start_val':first_guess[i],'limits':(0,nd)}

params_names,kwd = fitutils.dict2kwd(params_dict)

data_and_pdfs = [data,templates]

f = fitutils.Minuit_FCN([data_and_pdfs],params_dict,emlf_normalized_minuit)

m = minuit.Minuit(f,**kwd)

# For maximum likelihood method.
m.errordef = 0.5

# Up the tolerance.
#m.tol = 1.0

#m.print_level = 2
m.migrad(ncall=10000)
m.hesse()
#m.minos()

values = m.values

################################################################################
# Set up a figure for plotting.
################################################################################

fig0 = plt.figure(figsize=(9,6),dpi=100)
ax00 = fig0.add_subplot(1,1,1)
#ax01 = fig0.add_subplot(2,2,2)
#ax02 = fig0.add_subplot(2,2,3)
#ax03 = fig0.add_subplot(2,2,4)

'''
ax11.set_xlim(ranges[0],ranges[1])
ax00.plot(template0[0],values['num0']*template0[1],'g-',linewidth=3)
ax00.plot(template1[0],values['num1']*template1[1],'r-',linewidth=3)
ax00.plot(template1[0],values['num1']*template1[1]+values['num0']*template0[1],'b-',linewidth=3)
'''

binwidth = templates[0][0][0][1]-templates[0][0][0][0]
ax00.set_xlim(ranges[0],ranges[1])
ax00.plot(data[0][0],data[0][1],'ko')
for j in range(njets_min,njets_max+1):
    for i,s in enumerate(samples[0:-1]):
        tempx = templates[j-njets_min][i][0]-binwidth/2.0
        tempy = np.zeros(len(templates[0][0][1]))
        for k in range(i,5):
            name = "num_%s_njets%d" % (samples[k],j)
            tempy += values[name]*templates[j-njets_min][k][1]
        ax00.bar(tempx,tempy,color=fcolor[i],width=binwidth,edgecolor=fcolor[i])
        #plt.figure()
        #plt.bar(templates[j-njets_min][i][0]-binwidth/2.0,values[name]*templates[j-njets_min][i][1],color='b',width=binwidth,edgecolor='b')

plt.show()
print "ndata: "
print ndata


