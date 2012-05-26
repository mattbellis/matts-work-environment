import numpy as np
import matplotlib.pylab as plt
import scipy.constants as constants

def ls_form(x,c1,c2,R0,E0,r):

    #c1*R0*m/(E0*r)exp(-c2*ER/(E0*r))

    m = 7.0e30 # This isn't right.

    y = c1*R0*m/(E0*r)*np.exp(-c2*x/(E0*r))

    return y



'''
//going to turn lambdaNR and nNR into mass and cross section - use stuff from CoGeNT anaylysis
// DM signal - Parameterization of c1*R0*m/(E0*r)exp(-c2*ER/(E0*r)) for DM signal from PF Smith
// E0=1/2*MD*v0^2, where MD is the dark matter mass, and v0 is the galactic rotation velocity,
// E0 should be in the same units as the energy (keV), need to convert GeV to keV and convert v0 into units of c
// r=4MD*MT/(MD+MT)^2, where MT is the mass of the target in the same units as the dark matter->r is unitless
// R0, total event rate per unit mass, R0=2*N0*rhoD*sigma0*v0/(Sqrt(Pi)*AW*MD),
// AW is the target mass in AMU, AW=MT/pmass, N0 is Avogadro's Number,
// rhoD is the mass density of the DM ~0.3GeV/c^2/cm^3, sigmaSIp is the proton cross section in cm^2~ 10^-40,
// need to convert v0 to cm/s to make everything work out
// sigma0=sigmaSIp*AW^2*muTD^2/muPD^2, where muTD is the reduced mass of dark matter and target,
// and muPD is the reduced mass of the dark matter and proton muAB=MA*MB/(MA+MB)
// in the end R0 is in counts/s/g, multiply by number of seconds per day for total pdf to be in counts/day/keV
// m is active detector mass in grams
// c1=0.751 and c2=0.561 averaged over the year
'''

N0 = constants.N_A # Avogadro's number
c_ms = constants.c # Speed of light in m/s
pi = np.pi
seconds_per_day = 24.0*60.0*60.0

c1 = 0.751
c2 = 0.561

mass_dm = 7.0 # GeV/c^2

mass_Ge_amu = 72.63 # amu 0
mass_p_amu = 1.00727
amu_to_GeV = 0.9315

mass_Ge = mass_Ge_amu*amu_to_GeV
mass_p = mass_p_amu*amu_to_GeV

target_mass_amu = mass_Ge_amu

r=4*mass_dm*mass_Ge/((mass_dm+mass_Ge)**2)

GeVtokeV = 1000000.0

# Reduced mass of the DM + target system
mu_dm_t = (mass_dm*mass_Ge)*(mass_dm + mass_Ge)
mu_dm_p = (mass_dm*mass_p)*(mass_dm + mass_p)

xsec_dm_p = 2e-41 # WIMP - proton cross section in cm^-2


rho = 0.3 # Local WIMP density in GeV/c^2/cm^3

v0 = 220.0 # km/s

E0= (1.0/2.0)*mass_dm*(v0**2)

#sigma0=sigmaSIp*AW^2*muTD^2/muPD^2,
sigma0 = xsec_dm_p*(target_mass_amu**2)*(mu_dm_p**2)/(mu_dm_t**2)

# R0=2*N0*rhoD*sigma0*v0/(Sqrt(Pi)*AW*MD)
R0 = 2*N0*rho*sigma0*v0/(np.sqrt(pi)*target_mass_amu*mass_dm)

print R0
print R0*seconds_per_day

# c1*R0*m/(E0*r)exp(-c2*ER/(E0*r))
x = np.linspace(0,1000000,1000)
y = ls_form(x,c1,c2,R0,E0,r)

plt.plot(x,y)

plt.show()







