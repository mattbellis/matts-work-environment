import numpy as np
import matplotlib.pylab as plt
import scipy.constants as constants

N0 = constants.N_A # Avogadro's number
c_ms = constants.c # Speed of light in m/s
pi = np.pi
seconds_per_day = 24.0*60.0*60.0

print "N0: %e" % (N0)
print "c: %e" % (c_ms)

################################################################################
# 
################################################################################
def ls_form(x,c1,c2,R0,E0,r):

    # x is ER, recoil energy

    #c1*R0*m/(E0*r)exp(-c2*ER/(E0*r))

    m = 330 # Mass of detctor in grams?

    E0r = E0*r
    print "E0: %f" % (E0)
    print "r: %f" % (r)
    print "E0r: %f" % (E0r)

    #R0 = 1.0

    y = (c1*R0*m/(E0r))*np.exp(-c2*x/(E0r))
    #y *= seconds_per_day*30 # For months
    #y *= seconds_per_day*442 # For CoGeNT running time

    return y

################################################################################
# 
################################################################################


'''
//going to turn lambdaNR and nNR into mass and cross section - use stuff from CoGeNT anaylysis
// DM signal - Parameterization of c1*R0*m/(E0*r)exp(-c2*ER/(E0*r)) for DM signal from PF Smith
// dR/dER
// E0=1/2*MD*v0^2, where 
// MD is the dark matter mass, and 
// v0 is the galactic rotation velocity,
// E0 should be in the same units as the energy (keV), need to convert GeV to keV and convert v0 into units of c
//
// r=4MD*MT/(MD+MT)^2, where 
// MT is the mass of the target in the same units as the dark matter->r is unitless
// R0, total event rate per unit mass, 

// R0=2*N0*rhoD*sigma0*v0/(Sqrt(Pi)*AW*MD),
// AW is the target mass in AMU, 
// AW=MT/pmass, 
// N0 is Avogadro's Number,
// rhoD is the mass density of the DM ~0.3GeV/c^2/cm^3, 
// sigmaSIp is the proton cross section in cm^2~ 10^-40,
// need to convert v0 to cm/s to make everything work out
//
// sigma0=sigmaSIp*AW^2*muTD^2/muPD^2, where 
// muTD is the reduced mass of dark matter and target,
// and muPD is the reduced mass of the dark matter and proton muAB=MA*MB/(MA+MB)
//
// in the end R0 is in counts/s/g, multiply by number of seconds per day for total pdf to be in counts/day/keV
// m is active detector mass in grams
// c1=0.751 and c2=0.561 averaged over the year
'''

################################################################################

c1 = 0.751
c2 = 0.561

mass_dm = 7.0 # GeV/c^2

mass_Ge_amu = 72.63 # amu 0
mass_p_amu = 1.00727
amu_to_GeV = 0.9315

mass_Ge = mass_Ge_amu*amu_to_GeV
mass_p = mass_p_amu*amu_to_GeV

target_mass_amu = mass_Ge_amu

r = 4*mass_dm*mass_Ge/((mass_dm+mass_Ge)**2)
print "mass DM: %f" % (mass_dm)
print "mass Ge: %f" % (mass_Ge)
print "r: %f" % (r)

GeVtokeV = 1000000.0

# Reduced mass of the DM + target system
mu_dm_t = (mass_dm*mass_Ge)/(mass_dm + mass_Ge)
mu_dm_p = (mass_dm*mass_p)/(mass_dm + mass_p)

#xsec_dm_p = 2e-41 # WIMP - proton cross section in cm^-2
xsec_dm_p = 1e-40 # WIMP - proton cross section in cm^-2

rho = 0.3 # Local WIMP density in GeV/c^2/cm^3
#rho /= 1e6 # Convert to GeV/c^2/m^3

#sigma0=sigmaSIp*AW^2*muTD^2/muPD^2,
sigma0 = xsec_dm_p*(target_mass_amu**2)*(mu_dm_p**2)/(mu_dm_t**2)
print "mass_Ge: %e" % (mass_Ge)
print "mass_p: %e" % (mass_p)
print "mass_dm: %e" % (mass_dm)
print "mu_dm_p: %e" % (mu_dm_p)
print "mu_dm_t: %e" % (mu_dm_t)
print "sigma_{SI}: %e" % (xsec_dm_p)
print "sigma_{0}: %e" % (sigma0)

mod_x = np.array([])
mod_y = np.array([])
roi_x = np.array([])
roi_y = np.array([])

fig0 = plt.figure(figsize=(12,8))
ax00 = fig0.add_subplot(1,2,1)
ax01 = fig0.add_subplot(1,2,2)

total_events = 0.0

for t in range(0,15,1):

    #v0 = 220.0 # km/s
    #v0 *= 1e3 # Convert to m/s
    v0 = 244.0 + 15*np.sin(2*pi*(t/12.0))
    v0 *= 1e3 # Convert to m/s

    E0 = (1.0/2.0)*mass_dm*(v0**2)/(c_ms*c_ms)
    E0 *= 1e6 # Convert GeV to keV 

    # R0=2*N0*rhoD*sigma0*v0/(Sqrt(Pi)*AW*MD)
    m_to_cm = 100.0 # Need this for units
    R0 = 2*N0*rho*sigma0*(v0*m_to_cm)/(np.sqrt(pi)*target_mass_amu*mass_dm)
    print "R0: %e" % (R0)
    #R0 *= seconds_per_day
    print "R0: %e" % (R0)
    #R0 = 1e20
    #print R0
    #print R0*seconds_per_day
    # c1*R0*m/(E0*r)exp(-c2*ER/(E0*r))

    x = np.linspace(0,10,1000)
    y = ls_form(x,c1,c2,R0,E0,r)

    #y *= seconds_per_day*30 # For events/month
    y *= seconds_per_day*442 # For events/exposure time

    bin_width = x[1]-x[0]
    print "bin_width: %f" % (bin_width)
    #y *= bin_width

    roi_lo = int(len(x)*0.05)
    roi_hi = int(len(x)*0.4)
    roi_x = x[roi_lo:roi_hi] # Region from 0.5 keV to 4.0 keV.
    roi_y = y[roi_lo:roi_hi] # Region from 0.5 keV to 4.0 keV.

    norm = y.sum()

    print "norm: %f" % (norm)

    mod_x = np.append(mod_x,t)
    mod_y = np.append(mod_y,roi_y.sum()*bin_width)

    total_events += roi_y.sum()*bin_width
    print "total events: %f" % (total_events)

    ax00.plot(x,y)
    ax01.plot(roi_x,roi_y)

print "total events: %f" % (total_events)

plt.figure()
plt.plot(mod_x,mod_y,'go')
plt.ylim(0.0,1.5*mod_y[2])

plt.show()







