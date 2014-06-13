import sys
import matplotlib.pylab as plt
import numpy as np
import csv
import itertools as it

import lichen.iminuit_fitting_utilities as fitutils

import iminuit as minuit

from plot_diffusion_data import read_in_a_microprobe_data_file,read_in_an_isotope_data_file

from scipy import interpolate


experiment = "FNDA1"
#experiment = "FNDA2"
element = "Fe"
#element = "Ni"

if len(sys.argv)>1:
    experiment = sys.argv[1]

if len(sys.argv)>2:
    element = sys.argv[2]

if experiment=="FNDA1":
    infile_mp = "new_data_021614/FNDA_1_microprobe.csv"
    infile_iso = "new_data_021614/FNDA_1_isotope.csv"
elif experiment=="FNDA2":
    infile_mp = "new_data_021614/FNDA_2_microprobe.csv"
    infile_iso = "new_data_021614/FNDA_2_isotope.dat"

################################################################################
# For the new files
################################################################################
#'''
# FNDA 1 
if experiment=="FNDA1":
    xis_offset = 0.000010
    #xis_offset = 0.00044 
    #xis_offset = 0.00050 
    hours = 120 
    #hours = 0.01 
    #### exp(-30.268 + 5.00 xFe - 13.39 xFe^2 + 6.30 xFe^3)
    D56_coeff = [-30.268,5.00,13.39,6.30]

    # Fe
    if element=="Fe":
        cmax0 = 0.0085 # fraction
        #cmin0 = 1.008 # fraction
        cmin0 = 0.99 # fraction
        light_isotope = 54.
        heavy_isotope = 56.
    # Ni
    elif element=="Ni":
        #cmin0 = 1-0.0085 # fraction
        #cmax0 = 1-0.985 # fraction
        cmin0 = 0.993 # fraction
        cmax0 = 0.01 # fraction
        light_isotope = 61.
        heavy_isotope = 62.


# FNDA 2
elif experiment=="FNDA2":
    # For Ni
    xis_offset = 0.0
    if element=="Ni":
        xis_offset = -0.000150
    hours = 96
    #### exp(-28.838 + 4.92 xFe - 12.91 xFe^2 + 6.17 xFe^3)
    D56_coeff = [-28.838,4.92,12.91,6.17]
    # Fe
    if element=="Fe":
        cmax0 = 0.01 # fraction
        cmin0 = 0.99 # fraction
        light_isotope = 54.
        heavy_isotope = 56.
    # Ni
    elif element=="Ni":
        cmax0 = 0.01 # fraction
        cmin0 = 0.99 # fraction
        #cmax0 = 0.009 # fraction
        #cmin0 = 0.991 # fraction
        light_isotope = 61.
        heavy_isotope = 62.



################################################################################
# Read in a data file
################################################################################
# Read in the microprobe data
xmp,ymp = read_in_a_microprobe_data_file(infile_mp,experiment,element)
xis,yis,yerris,cis = read_in_an_isotope_data_file(infile_iso,experiment,element)

print sys.argv[1].split('/')[-1].split('_micro')[0]
#exit()

if element=="Fe":
    yis *= 2 # Weird labeling in files (both FNDA 1 and FNDA 2) for Fe?.
if experiment=="FNDA1" and element=="Ni":
    yis += 14.37 # For Ni, FNDA 1

xis -= xis_offset
#xis = xis[::-1] # Do this for FNDA 1

#xmp *= -1 # Do this for FNDA 2
#####xis *= -1 # Do this for FNDA 2, NOT for Fe

#print xis
#print yis
#for index in [-13,-12,-11,-10,-9,-7]:
#for index in [-13,-12,-11,-10]:
#for index in []:
#if experiment=="FNDA1" and element=="Fe":
if len(sys.argv)>=4:
    for index in [int(sys.argv[3])]:
        xis = np.delete(xis,index)
        yis = np.delete(yis,index)
        yerris = np.delete(yerris,index)
        cis = np.delete(cis,index)

print xis
print yis
#yerris /= yerris
#yerris *= 5
#print yerris
#yerris *= 4
#exit()

print len(xis)
print len(yis)
#exit()


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
lo = xmp[0]
hi = xmp[-1]
interface_x = 0
npts = 100
dx   = (hi-lo)/float(npts)

xpos = np.linspace(lo,hi,npts) 

frac_interface = (interface_x-lo)/(hi-lo)
print "frac_interface: ",frac_interface

point_of_intial_interface = int(npts*frac_interface)
print "point_of_intial_interface: ", point_of_intial_interface
xvals = np.linspace(lo,hi,npts)

################################################################################
# Move in time
################################################################################
dt   = 600.0
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

        '''
        if t%10000==0:
            print "%d of %d" % (t,tmax)
        '''

        D56 = np.exp(D56_coeff[0] + (D56_coeff[1]*c56) - D56_coeff[2]*(c56**2) + D56_coeff[3]*(c56**3))

        # Condition for finite element approach to be stable. 
        if len( (D56*dt*invdx2)[D56*(dt*invdx2)>0.5])>0:
            print "D56*dt*invdx2: ",D56*(dt*invdx2)

        D54 = D56*((heavy_isotope/light_isotope)**mybeta) # For Fe
        #D54 = D56*((heavy_isotope/light_isotope)**(intercept + (mybeta*c56))) # CONCENTRATION DEPENDENT

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

    # FNDA CHANGE?
    # Do this only for Ni!!!!! ############################# Nickel
    #print c56
    if element=="Ni":
        c56 = 1.0 - c56
        c54 = 1.0 - c54

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
#params_dict['mybeta'] = {'fix':False,'start_val':0.0,'limits':(-1.0,1.0),'error':0.01} # FOR CONCENTRATION DEPENDENCE
#params_dict['intercept'] = {'fix':False,'start_val':0.5,'limits':(-1.0,1.0),'error':0.01} # FOR CONCENTRATION DEPENDENCE

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
##m.hesse()
m.minos()

#print m.get_fmin()
#exit()


values = m.values
errors = m.errors

print values
print errors

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
#plt.plot(xis,cis,'co')
#plt.plot(xis,yis,'go')

fig0 = plt.figure(figsize=(12,6))
fig0.add_subplot(1,1,1)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)

label = r"$^{%d}$%s simulation" % (heavy_isotope,element)
plt.plot(xpos,c56,'b-',linewidth=3,label=label)
plt.plot([interface_x,interface_x],[0,110.0])
plt.ylim(0,1.10)
plt.plot(xmp,ymp,'ro',label='microprobe data')
plt.plot(xis,cis,'co',label='data from isotope file')
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
plt.plot(xmp,ymp,'ro',label='microprobe data')
plt.ylabel('Concentration')
plt.xlabel('Meters')
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper right') # FNDA 2, NI
elif experiment=="FNDA2" and element=="Fe":
    plt.legend(loc='center left') # FNDA 2, Fe
else:
    plt.legend(loc='center right')
'''
name = "%s_%s_diffusion_profile.png" % (element,experiment)
plt.savefig(name)

# Plot the deltas
plt.figure(figsize=(12,6))
#plt.plot(xpos,(c56/c54 - 1.0)*1000.0,'o')
plotlabel = r"best fit $\delta$, $\beta$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
plt.plot(xpos,sim_deltas,'-',linewidth=3,label=plotlabel)
plt.errorbar(xis,yis,yerr=yerris,markersize=10,fmt='o',label=r'$\delta$ from isotope data')
#'''
plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (values['mybeta']+(2.0*errors['mybeta']))
plt.plot(xpos,sim_deltasfake,'-',label=plotlabel)
plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (values['mybeta']-(2.0*errors['mybeta']))
plt.plot(xpos,sim_deltasfake0,'-',label=plotlabel)
#plotlabel = r"simulated $\delta$, $\beta$=%3.2f" % (0.3)
#plt.plot(xpos,sim_deltasfake1,'-',label=plotlabel)
#'''
plt.ylim(-40,40) # FNDA 1, Fe
if element=="Ni":
    plt.ylim(-20,20) # FNDA 1, Ni
plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Meters',fontsize=24)
if experiment=="FNDA2" and element=="Ni":
    plt.legend(loc='upper left') # FNDA 2, NI
elif experiment=="FNDA1" and element=="Fe":
    plt.legend(loc='upper right') # FNDA 1, Fe
else:
    plt.legend(loc='upper left')
name = "%s_%s_delta.png" % (element,experiment)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
plt.savefig(name)

# Plot the deltas vs. concentration.
plt.figure(figsize=(12,6))
plotlabel = r"best fit $\delta$, $\beta$=%3.2f $\pm$ %4.3f" % (values['mybeta'],errors['mybeta'])
plt.plot(c56,sim_deltas,'-',linewidth=3,label=plotlabel)
label = r"$^{%d}$%s" % (heavy_isotope,element)
plt.errorbar(cis,yis,yerr=yerris,fmt='o',markersize=10,label=label)
plt.ylabel(r'$\delta$',fontsize=36)
plt.xlabel('Concentration',fontsize=24)
plt.legend(loc='upper left') # FNDA 2, NI
name = "%s_%s_delta_vs_concentration.png" % (element,experiment)
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.10)
plt.savefig(name)

#plt.figure()
#plt.plot(xis,cis,'o',label='data from isotope file')
#plt.legend()

print "Final value of beta: %f" % (values['mybeta'])

name = "%s_%s_output.csv" % (element,experiment)

f = open(name,'w')
#csv.writer(f).writerows(it.izip_longest(xmp,ymp,xis,cis,xpos,c56,c54,xis,yis,yerris,xpos,sim_deltas,sim_deltasfake,sim_deltasfake0,sim_deltasfake1))
csv.writer(f).writerows(it.izip_longest(xmp,ymp,xis,cis,xpos,c56,c54,xis,yis,yerris,xpos,sim_deltas,sim_deltasfake,sim_deltasfake0))

print xis
print yis

print "FVAL: %f" % (m.fval)
plt.show()

