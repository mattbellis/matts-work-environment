import sys
import matplotlib.pylab as plt
import numpy as np

import lichen.iminuit_fitting_utilities as fitutils

import iminuit as minuit

from plot_diffusion_data import read_in_a_microprobe_data_file,read_in_an_isotope_data_file

from scipy import interpolate



################################################################################
# Read in a data file
################################################################################
# Read in the microprobe data
xmp,ymp = read_in_a_microprobe_data_file(sys.argv[1])
xis,yis,yerris,cis = read_in_an_isotope_data_file(sys.argv[2])

# For FNDA 1
#print "XIS!!!!"
#print xis
xis -= 0.00126
#print xis
xis = xis[::-1]

#print xis



################################################################################
# mybeta
################################################################################
#mybeta = 0.25

################################################################################
# Spatial dimensions
################################################################################
#lo = -1155.0e-6
#hi = 1530.0e-6
lo = xmp[0]
hi = xmp[-1]
interface_x = 0
npts = 100
dx   = (hi-lo)/float(npts)

xpos = np.linspace(lo,hi,npts) 

cmax0 = 0.009 # fraction
cmin0 = 0.991 # fraction

frac_interface = (interface_x-lo)/(hi-lo)
print "frac_interface: ",frac_interface

point_of_intial_interface = int(npts*frac_interface)
print "point_of_intial_interface: ", point_of_intial_interface
xvals = np.linspace(lo,hi,npts)

################################################################################
# Move in time
################################################################################
dt   = 600.0
t0   = 0
hours = 24
tmax = (3600*hours) # seconds?

invdx2 = 1.0/(dx**2)

#D = 3.17e-10
#D = 3.17364e-10
#D = 3.185e-10

print "dx: ",dx
print "invdx2: ",invdx2
print "dt: ",dt

t = t0

tag = "default"

#imgcount = 0
#it = 0

################################################################################
# Calculate the deltas
################################################################################
def fitfunc(data,p,parnames,params_dict):

    print '----------------------------------'
    print ' Calculating diffusion profiles    '
    print '----------------------------------'

    pn = parnames
    #print "ere"
    #print pn
    #print p

    mybeta = p[pn.index('mybeta')]
    #print "mybeta: ",mybeta

    c56 = np.zeros(npts)
    c56[:point_of_intial_interface] = cmax0
    c56[point_of_intial_interface:] = cmin0
    c54 = c56.copy()

    t = t0
    while t<tmax:

        '''
        if t%10000==0:
            print "%d of %d" % (t,tmax)
        '''

        # FNDA1
        #exp(-30.268 + 5.00 xFe - 13.39 xFe^2 + 6.30 xFe^3)
        #D56 = np.exp(-30.268 + (5.00*c56) - 13.39*(c56**2) + 6.30*(c56**3))
        # FNDA2
        #exp(-28.838 + 4.92 xFe - 12.91 xFe^2 + 6.17 xFe^3)
        D56 = np.exp(-28.838 + (4.92*c56) - 12.91*(c56**2) + 6.17*(c56**3))

        # Condition for finite element approach to be stable. 
        if len( (D56*dt*invdx2)[D56*(dt*invdx2)>0.5])>0:
            print "D56*dt*invdx2: ",D56*(dt*invdx2)

        D54 = D56*((56.0/54.0)**mybeta)

        i = 0
        for D,concentration in zip([D56,D54],[c56, c54]):
            c01 = np.roll(concentration, 1) # This shifts things ``forward", so that this is the previous position.
            c10 = np.roll(concentration,-1) # This shifts things ``backward", so that this is the subsequent position.

            D01 = np.roll(D, 1)
            D10 = np.roll(D,-1)

            # All the points except the end points.
            term1 =  (D10-D01)*(c10-c01)
            term2 =  D*(c10-(2*concentration)+c01)
            concentration_temp = concentration + dt*(term1/(4*dx*dx) + term2/(dx*dx))

            # The end points # NOT RIGHT!!!!! BUT OK FOR NOW
            concentration_temp[0]  = concentration[0]  + (D[0]*(concentration[1] -concentration[0]))
            concentration_temp[-1] = concentration[-1] + (D[-2]*(concentration[-2]-concentration[-1]))

            # Copy it over. 
            if i==0:
                c56 = concentration_temp.copy()
            elif i==1:
                c54 = concentration_temp.copy()

            i += 1


        t += dt

    delta56_54 = (c56/c54 - 1.0)*1000.0

    return c56,c54,delta56_54

################################################################################
# Chi square function for minuit.
################################################################################
def chisq_minuit(data,p,parnames,params_dict):

    #print "EFORE CALL:"
    #print p
    #print parnames
    #print params_dict
    c56,c54,simulated_deltas = fitfunc(data,p,parnames,params_dict)

    print "values in fit: "
    print p
    print parnames

    xdata = data[0]
    ydata = data[1]
    yerrdata = data[2]

    # Interpolate between data points.
    predicted_delta = interpolate.interp1d(xpos,simulated_deltas)

    chi2 = 0.0
    #print xdata
    #print xpos
    for x,y,yerr in zip(xdata,ydata,yerrdata):
        #if x>=simulated_deltas[-1] and x<=simulated_deltas[0]:
        if x>=min(xpos) and x<=max(xpos):

            #print x,y,yerr
            #print "INTERPOLATE: ",predicted_delta(x)
            chi2 += ((predicted_delta(x)-y)**2)/(yerr**2)

    print "-------- CHI2: %f" % (chi2)

    return chi2




################################################################################
# Set up minuit
################################################################################
params_dict = {}
params_dict['mybeta'] = {'fix':False,'start_val':0.25,'limits':(0.10,1.0),'error':0.01}

params_names,kwd = fitutils.dict2kwd(params_dict,verbose=True)

# For chi-squared method
kwd['errordef'] = 1.0
kwd['print_level'] = 2

data = [xis,yis,yerris]

print kwd

f = fitutils.Minuit_FCN([data],params_dict,chisq_minuit)

m = minuit.Minuit(f,**kwd)

m.print_param()


m.migrad()
#m.hesse()

values = m.values

print values

final_values = []
final_values.append(values['mybeta'])
c56,c54,sim_deltas = fitfunc(data,final_values,['mybeta'],params_dict)

fake_values = []
fake_values.append(0.5)
c56fake,c54fake,sim_deltasfake = fitfunc(data,fake_values,['mybeta'],params_dict)

fake_values = []
fake_values.append(0.1)
c56fake0,c54fake0,sim_deltasfake0 = fitfunc(data,fake_values,['mybeta'],params_dict)

fake_values = []
fake_values.append(0.25)
c56fake1,c54fake1,sim_deltasfake1 = fitfunc(data,fake_values,['mybeta'],params_dict)




################################################################################
# Plot the result
################################################################################

fig0 = plt.figure(figsize=(14,4))
fig0.add_subplot(1,3,2)
plt.plot(xpos,c56,'o')
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(xmp,ymp,'o')
plt.plot(xis,cis,'o')

fig0.add_subplot(1,3,3)
plt.plot(xpos,c54,'o')
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(xmp,ymp)


# Plot the deltas
plt.figure()
#plt.plot(xpos,(c56/c54 - 1.0)*1000.0,'o')
plt.plot(xpos,sim_deltas,'-',linewidth=3)
plt.errorbar(xis,yis,yerr=yerris,markersize=10,fmt='o')
plt.plot(xpos,sim_deltasfake,'-')
plt.plot(xpos,sim_deltasfake0,'-')
plt.plot(xpos,sim_deltasfake1,'-')
plt.ylim(-20,20)

plt.figure()
plt.plot(xis,cis,'o')

plt.show()

