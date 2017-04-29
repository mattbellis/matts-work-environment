import sys
import matplotlib.pylab as plt
import numpy as np
import csv
import itertools as it

import lichen.iminuit_fitting_utilities as fitutils

import iminuit as minuit

from plot_diffusion_data_040415 import read_datafile

from scipy import interpolate

intercept = 0.0

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

infilename  = "new_data_040415/data_%s_%s_profile%s.dat" % (experiment,element,profile)
print(infilename)

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
    cmax0 = 0.99 # fraction
    cmin0 = 0.01 # fraction
    light_isotope = 60.
    heavy_isotope = 62.
################################################################################


#x,c0,c1,delta,delta_err,xoffset = read_datafile(infilename,experiment,element,profile)
#print x
#print c0
#exit()

c0 = 1.0
c1 = 1.0
x = np.arange(-0.0015,0.0015,0.000001)

conc = None
if element=="Fe":
    conc = c0
elif element=="Ni":
    conc = c1

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
lo = 2*x[0]
hi = 2*x[-1]
interface_x = 0
npts = 100
dx   = (hi-lo)/float(npts)

xpos = np.linspace(lo,hi,npts) 
print(lo,hi)
#exit()
#'''
#if element=="Ni":
if 0:
    xpos = xpos[::-1]
#'''
#'''
elif element=="Fe" and experiment=="FNDA1" and profile=="2":
    #xpos = xpos[::-1]
    xpos = -xpos
#'''

print(lo,hi)
print(xpos[0],xpos[-1])
#exit()

frac_interface = (interface_x-lo)/(hi-lo)
print("frac_interface: ",frac_interface)

point_of_intial_interface = int(npts*frac_interface)
print("point_of_intial_interface: ", point_of_intial_interface)
xvals = np.linspace(lo,hi,npts)

################################################################################
# Move in time
################################################################################
#dt   = 60.0
dt   = 300.0
#dt   = 1.0
#dt   = 60.0
t0   = 0
tmax = (3600*hours) # seconds?

invdx2 = 1.0/(dx**2)

#D = 3.17e-10
#D = 3.17364e-10
#D = 3.185e-10

print("dx: ",dx)
print("invdx2: ",invdx2)
print("dt: ",dt)

t = t0

#tag = "default"
tag = "dt300"

#imgcount = 0
#it = 0

################################################################################
# Calculate the deltas
################################################################################
def fitfunc(data,p,parnames,params_dict,flag=0):

    print('----------------------------------')
    print(' Calculating diffusion profiles    ')
    print('----------------------------------')

    pn = parnames
    #print "ere"
    #print pn
    #print p

    mybeta = p[pn.index('mybeta')]
    #intercept = p[pn.index('intercept')] # CONCENTRATION DEPENDENCE
    #intercept = 0.1
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
            print("D56*dt*invdx2: ",D56*(dt*invdx2))

        #################### CONC DEPENDENT BETA ###############################
        if flag==1:
            print("HEREEREERE")
            mybetamin = 0.1
            mybetamax = 0.5
            conc_norm = (c56-min(c56))/max(c56)
            #print conc_norm
            #mybeta = mybetamin + (mybetamax-mybetamin)*c56
            mybeta = mybetamax - (mybetamax-mybetamin)*c56
            #print mybeta
            #exit()
        ########################################################################

        # This is the normal beta
        print(mybeta)
        D54 = D56*((heavy_isotope/light_isotope)**mybeta) # For Fe

        #print D56
        #print mybeta
        #print heavy_isotope,light_isotope
        #exit()

        # Here, we'll interpret beta as E.
        #D54 = D56*(mybeta*(np.sqrt((heavy_isotope/light_isotope)) - 1.0) + 1) # For Fe
        #D54 = D56*((heavy_isotope/light_isotope)**(intercept + (mybeta*c56))) # CONCENTRATION DEPENDENT
        #print("intercept: ",intercept)

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

    return c56,c54,delta56_54,intercept
################################################################################
################################################################################

params_dict = {}
params_dict['mybeta'] = {'fix':False,'start_val':0.30,'limits':(0.05,2.0),'error':0.01}
#params_dict['offset'] = {'fix':False,'start_val':-0.000074,'limits':(-0.000200,0.000200),'error':0.0000001}
params_dict['offset'] = {'fix':False,'start_val':1.0e-8,'limits':(-0.00150,0.00150),'error':0.0000000001}

offset = 0.0


fake_values = []
fake_values.append(0.30)
data = [x,1.0,1.0]
c56s = []
c54s = []
sim_deltas = []
betas = [0.20,0.35,0.50]
for beta in betas:
    b = [beta]
    c56fake,c54fake,sim_deltasfake,intercept = fitfunc(data,b,['mybeta'],params_dict)
    c56s.append(c56fake)
    c54s.append(c54fake)
    sim_deltas.append(sim_deltasfake)

#betas.append(99)
#c56fake,c54fake,sim_deltasfake,intercept = fitfunc(data,[99],['mybeta'],params_dict,flag=1)
#c56s.append(c56fake)
#c54s.append(c54fake)
#sim_deltas.append(sim_deltasfake)


################################################################################
# Plot the result
################################################################################
# Convert meters back to microns
x *= 1e6
offset *= 1e6
xpos *= 1e6

# Plot the diffusion profiles.
fig0 = plt.figure(figsize=(12,6))
fig0.add_subplot(1,1,1)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)

#'''
# Don't put the offset in when plotting the diffusion profiles.
for beta,concentration in zip(betas,c56s):
    label = r"$^{%d}$%s simulation $\beta$=%.2f" % (heavy_isotope,element,beta)
    plt.plot(xpos-offset,concentration,'b-',linewidth=3,label=label)
plt.plot([interface_x,interface_x],[0,110.0]) # Draw line
plt.ylim(0,1.10)
#plt.plot(x,c0,'ro',label='Fe concentration data')
#plt.plot(x,c1,'bo',label='Ni concentration data')
print(x)
print(c0)
#exit()
plt.ylabel('Concentration',fontsize=24)
plt.xlabel('Microns',fontsize=24)
plt.legend()
#'''


'''
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper right') # FNDA 2, NI
elif experiment=="FNDA2" and element=="Fe":
    plt.legend(loc='center left') # FNDA 2, Fe
else:
    plt.legend(loc='lower left')
'''
#plt.legend(loc='lower left')

name = "%s_%s_%s_%s_diffusion_profile.png" % (element,experiment,profile,tag)
plt.savefig(name)

################################################################################
# Plot the deltas
################################################################################
scaling = 0.5 # From Heather
plt.figure(figsize=(12,6))
for beta,delta in zip(betas,sim_deltas):
    label = r"$^{%d}$%s simulation $\beta$=%.2f" % (heavy_isotope,element,beta)
    plt.plot(xpos-offset,delta*scaling,'-',label=label,linewidth=3)
#plt.xlim(-4000,2000)
plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Microns',fontsize=24)
plt.legend(loc='lower left')
print("intercept: ",intercept)
#name = "FAKE_COMPARISON_%s_%s_%s_%s_intercept_%.2f_delta.png" % (element,experiment,profile,tag,intercept)
name = "FAKE_COMPARISON_%s_%s_%s_%s_delta.png" % (element,experiment,profile,tag)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
plt.savefig(name)

plt.show()

