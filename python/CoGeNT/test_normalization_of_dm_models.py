import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import chris_kelso_code as dmm
import chris_kelso_code_mpmath as dmm_mp

import mpmath as mp

AGe = 72.6

mDM = 7.0

tc_SHM = dmm.tc(np.zeros(3))

################################################################################
def func(y,x):
    #tc_SHM = dmm.tc(np.zeros(3))
    dR = dmm.dRdErSHM(x,tc_SHM+y,AGe,mDM)
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

#gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: 459.0,epsabs=0.1)
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: 365.0)
print gdbl_int
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: 365.0,epsabs=0.01)
print gdbl_int
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: 365.0,epsabs=0.1)
print gdbl_int
gdbl_int = integrate.dblquad(func,lo,hi,lambda x: 1.0, lambda x: 365.0,epsabs=1.0)
print gdbl_int

#print gdbl_int[0]*(330.0/1000.0)
#print gdbl_int[0]*(330.0/1000.0)*0.867

f = lambda x,y: dmm_mp.dRdErSHM(x,tc_SHM+y,AGe,mDM)
mpquad = mp.quad(f, [lo,hi], [1, 365])
print "mpquad: ",mpquad




