import numpy as np
import matplotlib.pylab as plt

import lichen.iminuit_fitting_utilities as fitutils

import scipy.stats as stats

import iminuit as minuit

masses = [88.0, 87.0, 86.0, 84.0]

################################################################################
# Spatial dimensions
################################################################################
lo = 0
hi = 2775e-6
npts = 100

dt   = 0.5 # This needs to be small (0.01) or it doesn't work.
dx   = (hi-lo)/float(npts)
t0   = 0
tmax = (60*17) + 31.0

################################################################################
# Figure out where the boundary is wrt our individual
# spatial points.
################################################################################
frac_interface = hi/1520e-6

point_of_intial_interface = int(npts/frac_interface)

################################################################################
# Initial isotope information
################################################################################
#cmax0 = 6.13
#cmin0 = 0.62

pct_isotopes0 = np.array([83.05, 6.89, 9.537, 0.52])
#pct_isotopes0 = np.array([82.58, 7.0, 9.86, 0.56])
frac_isotopes0 = pct_isotopes0/100.0
print "frac_isotopes0:"
print frac_isotopes0



################################################################################
# Calculate the deltas
################################################################################
def fitfunc(data,p,parnames,params_dict):

    print '----------------------------------'
    print ' Calulating diffusion profiles    '
    print '----------------------------------'

    pn = parnames

    D = np.array([0.0,0.0,0.0,0.0])
    D[0] = p[pn.index('D88')]
    D[1] = p[pn.index('D87')]
    D[2] = p[pn.index('D86')]
    D[3] = p[pn.index('D84')]
    cmax0 = p[pn.index('cmax0')]
    cmin0 = p[pn.index('cmin0')]
    #cmax0 = 6.13
    #cmin0 = 0.62

    xpos = np.linspace(lo,hi,npts)

    concentration = np.zeros(npts)
    concentration[:point_of_intial_interface] = cmax0
    concentration[point_of_intial_interface:] = cmin0

    # Declare and initialize the concentrations
    c = []
    c01 = []
    c10 = []

    bulk_c = np.zeros(npts)
    bulk_c01 = np.zeros(npts)
    bulk_c10 = np.zeros(npts)

    frac10 = np.zeros(npts)
    frac01 = np.zeros(npts)

    # Initialize everything
    for i in range(4):

        c.append(np.zeros(npts))
        c01.append(np.zeros(npts))
        c10.append(np.zeros(npts))

        c[i][:point_of_intial_interface] = cmax0*frac_isotopes0[i]
        c[i][point_of_intial_interface:] = cmin0*frac_isotopes0[i]

        c01.append(np.zeros(npts))
        c10.append(np.zeros(npts))

    bulk_c = c[0] + c[1] + c[2] + c[3]

    invdx2 = 1.0/(dx**2)

    print "dx: %f\tinvdx2: %f\tdt: %f" % (dx,invdx2,dt)
    Dnum = dt*invdx2
    Dnum *= D
    print "D*dt*invdx2: ",Dnum
    print "Dnum should be less than 1/2 for stability."
    # http://www.me.ucsb.edu/~moehlis/APC591/tutorials/tutorial5/node3.html

    t = t0

    print "tmax: ",tmax
    while t<tmax:

        '''
        if (t%100)<=dt:
            print t
        '''

        #print "bulk: ----------- "
        bulk_c = c[0] + c[1] + c[2] + c[3]

        # Loop over all 4 isotopes and propogate them individually.
        for i in range(4):

            frac = c[i]/bulk_c

            # Roll over the others to speed up the calculation
            c01[i] = np.roll(c[i], 1) # The i-1 val
            c10[i] = np.roll(c[i],-1) # The i+1 val

            ########################################################################
            # Using the individual species' concentration
            ########################################################################
            concentration_temp = c[i] + Dnum[i]*(c01[i]-c[i] + c10[i]-c[i])
            # The end points
            concentration_temp[0]  = c[i][0]  + Dnum[i]*((c[i][1] - c[i][0]))
            concentration_temp[-1] = c[i][-1] + Dnum[i]*((c[i][-2]-c[i][-1]))

            # Copy over the temporary array
            c[i] = concentration_temp.copy()

        t += dt

    deltas = [None,None,None]

    for i in range(1,4):
        deltas[i-1] = ((c[i]/frac_isotopes0[i]) / (c[0]/frac_isotopes0[0]) - 1.0)*1000.0

    return xpos,deltas


################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def chisq_minuit(data,p,parnames,params_dict):

    fit_x,fit_deltas = fitfunc(data,p,parnames,params_dict)

    print p

    chisq = 0.0
    for i in range(0,3):
        for x,y,yerr in zip(data[0],data[2*i+1],data[(2*i)+2]):
            if x<hi:
                # Find the indices of the closest points
                #print fit_x
                #print "x: ",x,fit_x.searchsorted(x)
                ihi = np.argmin(np.abs(fit_x-x))
                ilo = ihi-1
                if fit_x[ihi]-x<0:
                    ilo = ihi
                    ihi = ilo + 1

                #print "here"
                y0 = fit_deltas[i][ilo]
                y1 = fit_deltas[i][ihi]
                x0 = fit_x[ilo]
                x1 = fit_x[ihi]
                slope = (y1-y0)/(x1-x0)
                fit_y = (x-x0)*slope + y0
                #print "--------------"
                #print x,y
                #print ilo,ihi
                #print x0,x1,y0,y1,fit_y
                #print "HERE"
                #print fit_y
                chisq += ((fit_y-y)**2)/(yerr**2)

    print "chisq: ",chisq
    return chisq

################################################################################




################################################################################
# Importing the diffusion data.
################################################################################
# Full path to the directory 
infile_name = 'data.dat'
infile = open(infile_name)

content = np.array(infile.read().split()).astype('float')
print content

ncols = 7
ncontent = len(content)
print ncontent

delta = [None,None,None]
delta_err = [None,None,None]

index = np.arange(0,ncontent,ncols)
print index
xpos = content[index]/1e6
print xpos
delta[0] = content[index+1]
delta_err[0] = content[index+2]
delta[1] = content[index+3]
delta_err[1] = content[index+4]
delta[2] = content[index+5]
delta_err[2] = content[index+6]

############################################################################
# Plot the data
############################################################################
fig0 = plt.figure(figsize=(12,4),dpi=100)
axes0 = []
data = [xpos]
for i in range(0,3):
    axes0.append(fig0.add_subplot(1,3,i+1))
    plt.errorbar(xpos*1e6,delta[i],yerr=delta_err[i],fmt='o')
    plt.xlabel(r'Position ($\mu$ m)')
    plt.ylabel(r'$\delta$')
    data.append(delta[i])
    data.append(delta_err[i])
plt.subplots_adjust(top=0.92,bottom=0.15,right=0.95,left=0.10,wspace=0.2,hspace=0.35)

#plt.show()
#exit()


############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}
params_dict['D88'] = {'fix':True,'start_val':3.17e-10,'limits':(3.1e-10,3.5e-10)}
params_dict['D87'] = {'fix':False,'start_val':3.1718e-10,'limits':(3.1e-10,4.0e-10)}
params_dict['D86'] = {'fix':False,'start_val':3.1736e-10,'limits':(3.1e-10,4.0e-10)}
params_dict['D84'] = {'fix':False,'start_val':3.198-10,'limits':(3.1e-10,7.0e-10)}
params_dict['cmax0'] = {'fix':False,'start_val':6.13,'limits':(3.0,7.0)}
params_dict['cmin0'] = {'fix':False,'start_val':0.6,'limits':(0.1,3.0)}
#cmax0 = 6.13
#cmin0 = 0.62

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([data],params_dict,chisq_minuit)

m = minuit.Minuit(f,**kwd)

# For maximum likelihood method.
#m.errordef = 0.5
# For chi-squared method
m.errordef = 1.0

# Up the tolerance.
m.tol = 1.0

m.migrad()

values = m.values

print values
final_values = []
final_values.append(values['D88'])
final_values.append(values['D87'])
final_values.append(values['D86'])
final_values.append(values['D84'])
final_values.append(values['cmax0'])
final_values.append(values['cmin0'])


fit_x,fit_deltas = fitfunc(data,final_values,['D88','D87','D86','D84','cmax0','cmin0'],params_dict)

for i in range(0,3):
    axes0[i].plot(fit_x*1e6,fit_deltas[i])



plt.show()


