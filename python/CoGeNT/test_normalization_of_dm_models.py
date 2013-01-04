import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import chris_kelso_code as dmm
#import chris_kelso_code_mpmath as dmm_mp

#import mpmath as mp

AGe = 72.6

mDM = 7.0

tc_SHM = dmm.tc(np.zeros(3))

sigma_n = 1e-40

# For a stream with a maximum velocity
vMaxMod = np.zeros(3)
vEWinter = np.zeros(3)

dmm.vE_t(vEWinter,tc_SHM+365./2)
vMaxMod = dmm.normalize(vEWinter)
tc_Max = dmm.tc(vMaxMod)

############################################################################
#Now Choose a maximum modulating stream that has a cut-off at the given energy
############################################################################
#Er1=3.5
Er1=3
v01=10 #v01 is the dispersion of this stream
vstr1 = dmm.vstr(vMaxMod,AGe,tc_SHM,mDM,Er1)
print "\nStream characteristics for a target with atomic number %.2f and energy cut-off at %f keV:" % (AGe,Er1)
vstr1 = dmm.vstr(vMaxMod,AGe,153,mDM,Er1)
vstr1Vec = np.array([vstr1*vMaxMod[0],vstr1*vMaxMod[1],vstr1*vMaxMod[2]])



################################################################################
def func(y,x):
    #tc_SHM = dmm.tc(np.zeros(3))
    #dR = dmm.dRdErSHM(x,tc_SHM+y,AGe,mDM,sigma_n)
    dR = dmm.dRdErStream(x, tc_Max+y, AGe, vstr1Vec, 10,mDM,sigma_n)
    #dR = dmm.dRdErDebris(x,tc_Max+y,AGe,mDM,vDeb1,sigma_n)
    return dR
################################################################################

#Er = np.linspace(0.5,5.0,100)

#dR = dmm.dRdErSHM(Er, tc_SHM, AGe, mDM)

#plt.plot(Er,dR)

#lo = 0.0
#hi = 20.
#lo = 0.5
#hi = 3.2
#lo = 1.0
#hi = 3.0
lo = dmm.quench_keVee_to_keVr(0.5)
hi = dmm.quench_keVee_to_keVr(3.2)
#lo = dmm.quench_keVr_to_keVee(0.5)
#hi = dmm.quench_keVr_to_keVee(3.2)

print lo,hi

last_day = 459.0
#gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: last_day)
#print gdbl_int
#gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: last_day,epsabs=0.01)
#print gdbl_int
#gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: last_day,epsabs=0.1)
#print gdbl_int
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: last_day,epsabs=1.0)
print gdbl_int
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: last_day,epsabs=10.0)
print gdbl_int

#print gdbl_int[0]*(330.0/1000.0)
#print gdbl_int[0]*(330.0/1000.0)*0.867

#f = lambda x,y: dmm_mp.dRdErSHM(x,tc_SHM+y,AGe,mDM)
#mpquad = mp.quad(f, [lo,hi], [1, 365])
#print "mpquad: ",mpquad




