import sys
import matplotlib.pylab as plt
import numpy as np
import csv
import itertools as it

import lichen.iminuit_fitting_utilities as fitutils

import iminuit as minuit

from plot_diffusion_data_040415 import read_datafile

from scipy import interpolate


experiment = "FNDA2"
#experiment = "FNDA1"

element = "Fe"
#element = "Ni"

profile = "1"
#profile = "2"
#profile = "3"

if len(sys.argv)>1:
    experiment = sys.argv[1]

if len(sys.argv)>2:
    element = sys.argv[2]

if len(sys.argv)>3:
    profile = sys.argv[3]

infilename  = "new_data_040415/data_%s_%s_profile%s.dat" % (experiment,element,profile)
print infilename

################################################################################
if experiment=="FNDA1":
    hours = 120
    #### exp(-30.268 + 5.00 xFe - 13.39 xFe^2 + 6.30 xFe^3)
    D56_coeff = [-30.268,5.00,13.39,6.30]

elif experiment=="FNDA2":
    hours = 96
    D56_coeff = [-28.838,4.92,12.91,6.17]


if element=="Fe":
    cmax0 = 0.99 # fraction
    cmin0 = 0.01 # fraction
    light_isotope = 54.
    heavy_isotope = 56.
# Ni
elif element=="Ni":
    cmax0 = 0.01 # fraction
    cmin0 = 0.99 # fraction
    light_isotope = 60.
    heavy_isotope = 62.
################################################################################


x,c0,c1,delta,delta_err,xoffset = read_datafile(infilename,experiment,element,profile)

conc = None
if element=="Fe":
    conc = c0
    if experiment=="FNDA1" and profile=="2":
        x = x[::-1]
elif element=="Ni":
    conc = c1
    #x *= -1.0
    x = x[::-1]

################################################################################
# mybeta
################################################################################
#mybeta = 0.25

################################################################################
# Spatial dimensions
################################################################################
#lo = -1155.0e-6
#hi = 1530.0e-6
#xmp += 0.000060
lo = x[0]
hi = x[-1]
interface_x = 0
npts = 100
dx   = (hi-lo)/float(npts)

xpos = np.linspace(lo,hi,npts) 
if element=="Ni":
    xpos = xpos[::-1]
elif element=="Fe" and experiment=="FNDA1" and profile=="2":
    xpos = xpos[::-1]

print lo,hi
print xpos[0],xpos[-1]
#exit()

frac_interface = (interface_x-lo)/(hi-lo)
print "frac_interface: ",frac_interface

point_of_intial_interface = int(npts*frac_interface)
print "point_of_intial_interface: ", point_of_intial_interface
xvals = np.linspace(lo,hi,npts)

################################################################################
# Move in time
################################################################################
dt   = 60.0
#dt   = 1.0
#dt   = 60.0
t0   = 0
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
    #intercept = p[pn.index('intercept')] # CONCENTRATION DEPENDENCE
    #print "mybeta: ",mybeta

    c56 = np.zeros(npts)
    c56[:point_of_intial_interface] = cmax0
    c56[point_of_intial_interface:] = cmin0

    c54 = c56.copy()

    t = t0
    while t<tmax:

        #print t
        #print c56
        #print c54

        '''
        if t%10000==0:
            print "%d of %d" % (t,tmax)
        '''

        #print "COEFF"
        D56 = np.exp(D56_coeff[0] + (D56_coeff[1]*c56) - D56_coeff[2]*(c56**2) + D56_coeff[3]*(c56**3))

        # Condition for finite element approach to be stable. 
        if len( (D56*dt*invdx2)[D56*(dt*invdx2)>0.5])>0:
            print "D56*dt*invdx2: ",D56*(dt*invdx2)

        # This is the normal beta
        D54 = D56*((heavy_isotope/light_isotope)**mybeta) # For Fe

        #print D56
        #print mybeta
        #print heavy_isotope,light_isotope
        #exit()

        # Here, we'll interpret beta as E.
        #D54 = D56*(mybeta*(np.sqrt((heavy_isotope/light_isotope)) - 1.0) + 1) # For Fe
        #D54 = D56*((heavy_isotope/light_isotope)**(intercept + (mybeta*c56))) # CONCENTRATION DEPENDENT

        #print "DSSS",D56,D54


        i = 0
        for D,concentration in zip([D56,D54],[c56, c54]):
            c01 = np.roll(concentration, 1) # This shifts things ``forward", so that this is the previous position.
            c10 = np.roll(concentration,-1) # This shifts things ``backward", so that this is the subsequent position.

            D01 = np.roll(D, 1)
            D10 = np.roll(D,-1)

            # All the points except the end points.
            term1 =  (D10-D01)*(c10-c01)
            term2 =  D*(c10-(2*concentration)+c01)
            #print "TERM1"
            #print term1
            #print "TERM2"
            #print term2
            #print "DTETC"
            #print dt,dx
            #print dt*(term1/(4*dx*dx) + term2/(dx*dx))
            #print "CONCENTRATION"
            #print concentration
            concentration_temp = concentration + dt*(term1/(4*dx*dx) + term2/(dx*dx))

            # The end points # NOT RIGHT!!!!! BUT OK FOR NOW
            concentration_temp[0]  = concentration[0]  + (D[0]*(concentration[1] -concentration[0]))
            concentration_temp[-1] = concentration[-1] + (D[-2]*(concentration[-2]-concentration[-1]))

            #print concentration_temp
            #print concentration_temp
            #exit()

            # Copy it over. 
            if i==0:
                c56 = concentration_temp.copy()
            elif i==1:
                c54 = concentration_temp.copy()

            i += 1


        t += dt

    # FNDA CHANGE?
    # Do this only for Ni!!!!! ############################# Nickel
    #print c56
    #'''
    if element=="Ni":
        c56 = 1.0 - c56
        c54 = 1.0 - c54
    #'''

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

    offset = p[parnames.index('offset')]

    # Interpolate between data points.
    predicted_delta = interpolate.interp1d(xpos,simulated_deltas)

    chi2 = 0.0
    #print xdata
    #print xpos
    for x,y,yerr in zip(xdata,ydata,yerrdata):
        #if x>=simulated_deltas[-1] and x<=simulated_deltas[0]:
        x += offset
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
params_dict['mybeta'] = {'fix':False,'start_val':0.30,'limits':(0.05,2.0),'error':0.01}
#params_dict['offset'] = {'fix':False,'start_val':-0.000074,'limits':(-0.000200,0.000200),'error':0.0000001}
params_dict['offset'] = {'fix':False,'start_val':1.0e-6,'limits':(-0.00200,0.00200),'error':0.00000001}
#params_dict['mybeta'] = {'fix':False,'start_val':0.0,'limits':(-1.0,1.0),'error':0.01} # FOR CONCENTRATION DEPENDENCE
#params_dict['intercept'] = {'fix':False,'start_val':0.5,'limits':(-1.0,1.0),'error':0.01} # FOR CONCENTRATION DEPENDENCE

params_names,kwd = fitutils.dict2kwd(params_dict,verbose=True)

# For chi-squared method
kwd['errordef'] = 1.0
kwd['print_level'] = 2

data = [x,delta,delta_err]

print "HERE"
print kwd

f = fitutils.Minuit_FCN([data],params_dict,chisq_minuit)

m = minuit.Minuit(f,**kwd)

print "THERE"
m.print_param()


m.migrad()
##m.hesse()
#m.minos()

#print m.get_fmin()
#exit()

values = m.values
errors = m.errors

print "Values:"
print values
print "Errors:"
print errors

mybeta = values['mybeta']
offset = values['offset']

final_values = []
final_values.append(values['mybeta'])
c56,c54,sim_deltas = fitfunc(data,final_values,['mybeta'],params_dict) 
#final_values.append(values['intercept']) # CONCENTRATION DEPENDENCE
#c56,c54,sim_deltas = fitfunc(data,final_values,['mybeta','intercept'],params_dict) # CONCENTRATION DEPENDENCE

#'''
fake_values = []
fake_values.append(values['mybeta']+(2.0*errors['mybeta']))
c56fake,c54fake,sim_deltasfake = fitfunc(data,fake_values,['mybeta'],params_dict)

fake_values = []
fake_values.append(values['mybeta']-(2.0*errors['mybeta']))
c56fake0,c54fake0,sim_deltasfake0 = fitfunc(data,fake_values,['mybeta'],params_dict)

#fake_values = []
#fake_values.append(0.3)
#c56fake1,c54fake1,sim_deltasfake1 = fitfunc(data,fake_values,['mybeta'],params_dict)

#'''



################################################################################
# Plot the result
################################################################################
fig0 = plt.figure(figsize=(12,6))
fig0.add_subplot(1,1,1)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)

label = r"$^{%d}$%s simulation" % (heavy_isotope,element)
plt.plot(xpos,c56,'b-',linewidth=3,label=label)
plt.plot([interface_x,interface_x],[0,110.0]) # Draw line
plt.ylim(0,1.10)
plt.plot(x,c0,'ro',label='Fe concentration data')
plt.plot(x,c1,'bo',label='Ni concentration data')
plt.ylabel('Concentration',fontsize=24)
plt.xlabel('Meters',fontsize=24)
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper right') # FNDA 2, NI
elif experiment=="FNDA2" and element=="Fe":
    plt.legend(loc='center left') # FNDA 2, Fe
else:
    plt.legend(loc='center right')

'''
fig0.add_subplot(1,2,2)
label = r"$^{%d}$%s simulation" % (light_isotope,element)
#plt.plot(xpos,c54,'b-',label='Fe54 simulation ')
plt.plot(xpos,c54,'b-',linewidth=3,label=label)
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(x,conc,'ro',label='microprobe data')
plt.ylabel('Concentration')
plt.xlabel('Meters')
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper right') # FNDA 2, NI
elif experiment=="FNDA2" and element=="Fe":
    plt.legend(loc='center left') # FNDA 2, Fe
else:
    plt.legend(loc='center right')
'''
name = "%s_%s_%s_%s_diffusion_profile.png" % (element,experiment,profile,tag)
plt.savefig(name)

# Plot the deltas
plt.figure(figsize=(12,6))
#plt.plot(xpos,(c56/c54 - 1.0)*1000.0,'o')
plotlabel = r"best fit $\delta$, $\beta$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
#plotlabel = r"best fit $\delta$, $E$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
plt.plot(xpos,sim_deltas,'-',linewidth=3,label=plotlabel)
plt.errorbar(x+offset,delta,yerr=delta_err,markersize=10,fmt='o',label=r'$\delta$ from isotope data')
#'''
plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (values['mybeta']+(2.0*errors['mybeta']))
#plotlabel = r"simulated $\delta$, $E$=%3.2f" % (values['mybeta']+(2.0*errors['mybeta']))
plt.plot(xpos,sim_deltasfake,'-',label=plotlabel)
#plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (values['mybeta']-(2.0*errors['mybeta']))
plotlabel = r"simulated $\delta$, $E$=%3.2f" % (values['mybeta']-(2.0*errors['mybeta']))
plt.plot(xpos,sim_deltasfake0,'-',label=plotlabel)
#plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (0.3)
#plt.plot(xpos,sim_deltasfake1,'-',label=plotlabel)
#'''
'''
plt.ylim(-40,40) # FNDA 1, Fe
if element=="Ni":
    plt.ylim(-20,20) # FNDA 1, Ni
'''
plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Meters',fontsize=24)
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper left') # FNDA 2, NI
elif experiment=="FNDA1" and element=="Fe":
    plt.legend(loc='upper right') # FNDA 1, Fe
else:
    plt.legend(loc='upper left')
name = "%s_%s_%s_%s_delta.png" % (element,experiment,profile,tag)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
plt.savefig(name)

# Plot the deltas vs. concentration.
plt.figure(figsize=(12,6))
plotlabel = r"best fit $\delta$, $\beta$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
#plotlabel = r"best fit $\delta$, $E$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
plt.plot(c56,sim_deltas,'-',linewidth=3,label=plotlabel)
label = r"$^{%d}$%s" % (heavy_isotope,element)
plt.errorbar(conc,delta,yerr=delta_err,fmt='o',markersize=10,label=label)
plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Concentration',fontsize=24)
plt.legend(loc='upper left') # FNDA 2, NI
name = "%s_%s_%s_%s_delta_vs_concentration.png" % (element,experiment,profile,tag)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
plt.savefig(name)

#plt.figure()
#plt.plot(x,conc,'o',label='data from isotope file')
#plt.legend()

print "Final value of beta: %f" % (values['mybeta'])

name = "%s_%s_%s_%s_output.csv" % (element,experiment,profile,tag)

f = open(name,'w')
#csv.writer(f).writerows(it.izip_longest(x,conc,x,conc,xpos,c56,c54,x,delta,delta_err,xpos,sim_deltas,sim_deltasfake,sim_deltasfake0,sim_deltasfake1))
csv.writer(f).writerows(it.izip_longest(x,conc,x,conc,xpos,c56,c54,x,delta,delta_err,xpos,sim_deltas,sim_deltasfake,sim_deltasfake0))


print experiment,element
print "mybeta: %f" % (mybeta)
print "offset: %f" % (offset)
print "FVAL: %f" % (m.fval)
plt.show()

