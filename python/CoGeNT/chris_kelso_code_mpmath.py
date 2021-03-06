import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import mpmath 

################################################################################
hbarc = 0.1973269664036767; # in GeV nm 
c = constants.c*100.0 # cm/s 
rhoDM = 0.3;  # in GeV/cm^3
mDM = 6.8; # in GeV 
#mDM = 8.8; # in GeV 
mn = 0.938; # mass of nucleon in GeV 
mU = 0.9315; # amu in GeV 
sigma_n = 1E-40; # in cm^2 
#sigma_n = 0.5e-40; # in cm^2 
#Na = 6.022E26; # Avogadro's # in mol/kg 
Na = constants.N_A*1000.0 # Avogadro's # in mol/kg 
M_PI = np.pi

################################################################################
# double zeroVec[]={0,0,0};
# Atomic masses of targets
################################################################################
AGe = 72.6
ANa = 23
AXe = 131.6
################################################################################

################################################################################
def dot(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

################################################################################
# this function will normalize first vector and put in second vector
################################################################################
def normalize(v):
    vnorm = np.zeros(3)
    for i,vx in enumerate(v):
        vnorm[i] = vx/mpmath.sqrt(dot(v,v))

    return vnorm


################################################################################
# Conversion
################################################################################
def mT(A):
    return A*mU


def mu_n(m):
    return (m*mn)/(m + mn)


def mu(A, m):
    return (mT(A)*m)/(mT(A)+m)

################################################################################

################################################################################
# Quenching factor
################################################################################
def quench_keVr_to_keVee(x):
    y = 0.2*(x**(1.12))
    return y

def quench_keVee_to_keVr(x):
    y = (x/0.2)**(1.0/1.12)
    return y


################################################################################
#minimum speed DM must have to produce recoil of energy Er
# Er is assumed to be in keV, and A is atomic mass of target, m is DM mass
# result is in km/s
################################################################################
def vmin(Er, A, m):
    #return c*mpmath.sqrt(Er*mT(A)/2)*1e-8/mu(A,m)
    ret = c*mpmath.sqrt(Er*mT(A)/2)*1e-8/mu(A,m)
    #print ret
    if type(ret)==np.ndarray:
        # Remove nans
        ret[ret!=ret] = 0
    return ret

################################################################################
#Earth orbit parameters
################################################################################
t1 = 0.218*365 #  
e1 = np.array([0.9931,0.1170,-0.01032])
e2= np.array([-0.0670, 0.4927, -0.8676])
vorb = 30
omega = 2*M_PI/365.0


################################################################################
#SHM parameters      #
################################################################################
vo = 220
vesc = 544
# galactic velocity of sun, comes from peculiar velocity and local standard of rest 
vsVec = np.array([10,13+vo,7])
vs = np.sqrt(dot(vsVec,vsVec))
vsVecHat = np.array([vsVec[0]/vs,vsVec[1]/vs,vsVec[2]/vs])
z = vesc/vo
Nesc = mpmath.erf(z)-2*z*mpmath.exp(-z*z)/mpmath.sqrt(M_PI)



################################################################################
# this function will put the total velocity of the earth (observer) in galactic coordinates in the passed vector at the given time
################################################################################
def vObs_t(vObs, t):
    for i in range(0,3):
        #print  (vorb*(mpmath.cos(omega*(t-t1))*e1[i]+mpmath.sin(omega*(t-t1))*e2[i])) 
        #print  (vorb*(mpmath.cos(omega*(t-t1))*e1[i]+mpmath.sin(omega*(t-t1))*e2[i])) + vsVec[i]
        #print  type((vorb*(mpmath.cos(omega*(t-t1))*e1[i]+mpmath.sin(omega*(t-t1))*e2[i])) + vsVec[i])
        #print type(vObs[i])
        vObs[i] = vsVec[i]+vorb*(mpmath.cos(omega*(t-t1))*e1[i]+mpmath.sin(omega*(t-t1))*e2[i])


################################################################################
# this function will put only the orbital velocity of the earth in galactic coordinates in the passed vector at the given time
################################################################################
def vE_t(vE, t):
    for i in range(0,3):
        vE[i] = vorb*(mpmath.cos(omega*(t-t1))*e1[i]+mpmath.sin(omega*(t-t1))*e2[i])

################################################################################
# this function is the projection of the observer's velocity onto the stream
# the veoctr that is passes must be the unit vecotr pointing along the stream
# since the earth moves, this function is time dependent
################################################################################
def alpha(vstr, t):
    vObs = np.zeros(3)
    vObs_t(vObs,t)
    return dot(vObs,vstr)


################################################################################
# this function will give the magnitude of the stream along the given direction
# that will have a cut-off at the given recoil energy for target atomic number A
# at a time t and for dark matter mass of m 
################################################################################
def vstr(vstrHat, A, t, m, Er):
    vObs = np.zeros(3)
    vObs_t(vObs,t)
    vObsSqrd = dot(vObs,vObs)
    return alpha(vstrHat, t)+mpmath.sqrt(alpha(vstrHat, t)*alpha(vstrHat, t)+vmin(Er,A,m)*vmin(Er,A,m)-vObsSqrd)


################################################################################
# this function gives the speed of the stream in the frame of the earth
################################################################################
def vstrEarth(vstr, t):
    vObs = np.zeros(3)
    vObs_t(vObs,t)
    return mpmath.sqrt(dot(vstr,vstr)+dot(vObs,vObs)-2*dot(vstr,vObs))


################################################################################
# returns the energy cut-off for a given stream in keV
#t is in days, vstr in km/s, m is DM mass in GeV
################################################################################
def EcutOff(vstr,t,A,m):
    return 2.0e16*mu(A,m)*mu(A,m)*vstrEarth(vstr,t)*vstrEarth(vstr,t)/(c*c*mT(A))

#  return c*sqrt(Er*mT(A)/2)*1E-8/mu(A,m);

################################################################################
# This function finds the time when the stream will have its' max phase
################################################################################
def tc(vstr):
    b1 = 0.0
    b2 = 0.0
    b = 0.0
    t = 0.0
    vSunRel = np.array([vsVec[0]-vstr[0],vsVec[1]-vstr[1],vsVec[2]-vstr[2]])
    vSunRelHat = np.zeros(3)
    vSunRelHat = normalize(vSunRel)
    print "vSunRelHat: ",vSunRelHat
    b1 = dot(e1,vSunRelHat)
    b2 = dot(e2,vSunRelHat)
    b = mpmath.sqrt(b1*b1+b2*b2)
    t = mpmath.arccos(b1/b)/omega

    if(b2<0):
        t=2*M_PI/omega-t

    return t+t1

################################################################################
# Velocity distribution integral for a stream
# vstrE must be stream speed in frame of earth
# vo is the dispersion of the stream i.e. 
# f(v)~exp(-(v-vstrE)^2/v0^2)
################################################################################
def gStream(vmin, vstrE, v0):
    return (mpmath.erf((vmin+vstrE)/v0)-mpmath.erf((vmin-vstrE)/v0))/(2*vstrE)

################################################################################
# Same as above but if the stream has zero dispersion (i.e. delta function)
################################################################################
def gStreamZeroDispersion(vmin, vstrE):
    if(vmin>vstrE):
        return 0
    else:
        return 1/vstrE


################################################################################
#Thse functions are the velocities integrals for the SHM
################################################################################
def glow(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    y = mpmath.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    #  printf("\nve(t)=%f",vo*y);

    return (mpmath.erf(x+y)-mpmath.erf(x-y)-4*y*mpmath.exp(-z*z)/mpmath.sqrt(M_PI))/(2*Nesc*vo*y);


################################################################################
def ghigh(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    y = mpmath.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    return (mpmath.erf(z)-mpmath.erf(x-y)+2*(x-y-z)*mpmath.exp(-z*z)/mpmath.sqrt(M_PI))/(2*Nesc*vo*y)

################################################################################
def gSHM(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    y = mpmath.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    if type(Er)==np.ndarray:
        nvals = len(Er)
        ret = np.zeros(nvals)
        ret[x<z-y] = glow(Er,t,A,m)
        ret[x<z+y] = ghigh(Er,t,A,m)

        return ret
    else:
        if(x<z-y):
            return glow(Er,t,A,m)
        elif(x<y+z):
            return ghigh(Er,t,A,m)
        else:
            return 0




################################################################################
# This works ok for overall rates, but will not work well if doing a modulation analysis
################################################################################
def gDebris(vmin, vflow, t):
    vObs = np.zeros(3)
    vObs_t(vObs,t)
    vobs = mpmath.sqrt(dot(vObs,vObs))
    if(vmin<mpmath.abs(vflow-vobs)):
        return (vflow+vobs-mpmath.abs(vflow-vobs))/(2*vflow*vobs)
    elif(vmin<vflow+vobs):
        return (vflow+vobs-vmin)/(2*vflow*vobs)
    else:
        return 0

################################################################################
#Form Factor Functions
################################################################################
a=0.523
s=0.9

################################################################################
def cf(A):
    return 1.23*mpmath.power(A,1.0/3.0)-0.6



################################################################################
def q(Er, A):
    return mpmath.sqrt(2*mT(A)*Er*1.0E-6)

################################################################################
def R1(A):
    return mpmath.sqrt(cf(A)*cf(A)+7*M_PI*M_PI*a*a/3-5*s*s)

################################################################################
def Z1(Er,A):
    return q(Er,A)*R1(A)/hbarc

################################################################################
def Zs(Er, A):
    return q(Er,A)*s/hbarc

################################################################################
def j1(x):
    return (mpmath.sin(x)-x*mpmath.cos(x))/(x*x)

################################################################################
def F( Er,A):
    return 3*j1(Z1(Er,A))*mpmath.exp(-Zs(Er,A)*Zs(Er,A)/2)/Z1(Er,A);


################################################################################
#All of the particle physics in the spectrum that is not the velocity distribution integral
#This will give the spectra in counts per kg per keV per day 
################################################################################
def spectraParticle(Er,A, m):
    #return Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0e-11*24*3600/(2*m*mu_n(m)*mu_n(m))
    ret = Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0e-11*24*3600/(2*m*mu_n(m)*mu_n(m))
    if type(ret)==np.ndarray:
        # Remove nans
        ret[ret!=ret] = 0
    return ret


################################################################################
# The spectrum for a stream
################################################################################
def dRdErStream(Er, t, A, vstr, v0,m):
    return spectraParticle(Er,A,m)*gStream(vmin(Er,A,m),vstrEarth(vstr,t),v0)


################################################################################
# The SHM spectrum
################################################################################
def dRdErSHM(Er, t, A, m):
    return spectraParticle(Er,A,m)*gSHM(Er,t,A,m)


################################################################################
# The debris spectrum from Lisanti's paper
################################################################################
def dRdErDebris(Er,t,A,m,vflow):
    return spectraParticle(Er,A,m)*gDebris(vmin(Er,A,m),vflow,t)



################################################################################
# Main
################################################################################
def main():
    i = 0.0
    Er = 0.0

    ##############################################################################
    # Find the max phase for the SHM.  This corresponds to a stream with zero velocity

    tc_SHM = tc(np.zeros(3))
    print "\nThe SHM maximum phase occurs at %f days." % (tc_SHM)
    Er = np.linspace(0.5,5.0,100)
    dR = dRdErSHM(Er, tc_SHM, AGe, mDM)

    # Quenching factor?
    #Er = QF^-1 Eee
    #dRdEee=dRdEr*(dEr/dEee)
    Eee = Er/5

    plt.figure()
    #plt.plot(Er,dR)
    plt.plot(Eee,dR)
    #plt.show()

    output = "\nSpectrum for the SHM at t=%f\n" % (tc_SHM)
    for x,y in zip(Er,dR):
        print x,y

    plt.figure()
    ############################################################################
    # Make some plots
    ############################################################################
    npts = 100
    xvals = np.linspace(0.5,3.2,npts)
    yvals = np.zeros(npts)
    xt = np.linspace(0.0,11.0,12)
    yt = np.zeros(12)
    #for j in range(0,12):
    tot = 0.0
    month = 0
    lo = xvals[10]
    hi = xvals[60]
    norm_tot = 0.0
    print lo,hi
    for j in range(0,360):
        #day = 30*j
        day = j
        #tot += integrate.quad(dRdErSHM,lo,hi,args=(tc_SHM+day, AGe, mDM))[0]
        yvals = dRdErSHM(Er, tc_SHM+day, AGe, mDM)
        norm_tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        if (j+1)%30==0:

            yvals = dRdErSHM(Er, tc_SHM+day, AGe, mDM)
            plt.plot(xvals,yvals)

            yt[month] = tot
            tot = 0.0
            month += 1
            print month

    print "SHM: ",norm_tot
    plt.xlabel('Recoil energy (keVee)')
    #plt.show()

    plt.figure()
    plt.plot(xt,yt,'ro')

    #exit()

    ################################################################################
    #Now find the stream that has the maximum modulation
    #This is a stream that points opposite to the direction of the earth's orbital motion
    #at the time when the SHM has it's max. phase
    ################################################################################
    vMaxMod = np.zeros(3)
    vEWinter = np.zeros(3)

    vE_t(vEWinter,tc_SHM+365./2)
    vMaxMod = normalize(vEWinter)

    print "\n\n\n\nIn galactic coordinates, the stream with maximum modulation:\nDirection:"
    print vMaxMod
    tc_Max=tc(vMaxMod)
    print "\nMaximum phase at t=%f.\n" % (tc_Max)

    ############################################################################
    #Now Choose a maximum modulating stream that has a cut-off at the given energy
    ############################################################################
    Er1=3 
    v01=10 #v01 is the dispersion of this stream
    vstr1 = vstr(vMaxMod,AGe,tc_SHM,mDM,Er1)
    print "\nStream characteristics for a target with atomic number %.2f and energy cut-off at %f keV:" % (AGe,Er1)
    vstr1 = vstr(vMaxMod,AGe,153,mDM,Er1)
    vstr1Vec = np.array([vstr1*vMaxMod[0],vstr1*vMaxMod[1],vstr1*vMaxMod[2]])

    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vstr1,v01)

    print "\nIn earth's frame,";  

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (EcutOff(vstr1Vec, tc_Max,AGe, mDM),vstrEarth(vstr1Vec,tc_Max))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (EcutOff(vstr1Vec, tc_Max+365./2.,AGe, mDM),vstrEarth(vstr1Vec,tc_Max+365./2.))

    output = "Spectrum for this stream at t=%f\n" % (tc_Max)
    for i in range(0,11):
        Er=2.5+i*0.1
        output += "%.2E %.8E\n" % (Er,dRdErStream(Er, tc_Max, AGe, vstr1Vec, 10,mDM))
    print output



    #The Sagitarius stream may intersect the solar system
    vSag=300 
    v0Sag=10
    vSagHat = np.array([0,0.233,-0.970])
    vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
    tc_Sag = tc(vSagVec)

    print "\n\n\n\nThe Sagitarius Stream has a max. phase at %f.\n" % (tc_Sag)


    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vSag,v0Sag) 

    print "\nIn earth's frame,"

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (EcutOff(vSagVec, tc_Sag,AGe, mDM),vstrEarth(vSagVec,tc_Sag))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (EcutOff(vSagVec, tc_Sag+365./2.,AGe, mDM),vstrEarth(vSagVec,tc_Sag+365./2.))


    output = "Spectrum for this stream at t=%f\n" % (tc_Sag)
    for i in range(0,11):
        Er=1+i*0.1
        output += "%.2E %.8E\n" % (Er,dRdErStream(Er, t1, AGe, vSagVec, v0Sag,mDM))
    print output

    vDeb1 = 340
    print "\n\n\n\nDebris spectrum for %.1f\n" % (vDeb1)


    output = "\nSpectrum for debris at t=%f\n" % (tc_SHM)
    for i in range(0,11):
        Er = 0.5+i*0.5
        output += "%.2E %.8E\n" % (Er,dRdErDebris(Er, tc_SHM, AGe, mDM, vDeb1))
    print output

    ############################################################################
    # Make some plots
    ############################################################################
    plt.figure()
    npts = 100
    xvals = np.linspace(0.0,5.0,npts)
    yvals = np.zeros(npts)
    for j in range(0,12):
        day = 30*j
        for i,Er in enumerate(xvals):
            yvals[i] = dRdErDebris(Er, tc_SHM+day, AGe, mDM, vDeb1)

        #print yvals
        #print xvals
        #print integrate.trapz(yvals[1:-1],x=xvals[1:-1])
        #print integrate.simps(yvals[1:-1],x=xvals[1:-1])
        #print integrate.quad(dRdErDebris,xvals[1],xvals[-1],args=(tc_SHM+day, AGe, mDM, vDeb1))
        plt.plot(xvals,yvals)


    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()
