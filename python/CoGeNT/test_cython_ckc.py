import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import chris_kelso_code_cython as ckcc
#import chris_kelso_code as ckcc

################################################################################
hbarc = 0.1973269664036767; # in GeV nm 
#c = constants.c*100.0 # cm/s 
c = 3e10 # cm/s 
rhoDM = 0.3;  # in GeV/cm^3

#mDM = 6.8; # in GeV 
mDM = 7.0; # in GeV 
#mDM = 8.8; # in GeV 

mn = 0.938; # mass of nucleon in GeV 
mU = 0.9315; # amu in GeV 

#sigma_n = 1E-40; # in cm^2 
#sigma_n = 5E-40; # in cm^2 
#sigma_n = 0.5e-40; # in cm^2 
Na = 6.022E26; # Avogadro's # in mol/kg 
#Na = constants.N_A*1000.0 # Avogadro's # in mol/kg 
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
def main():
    i = 0.0
    Er = 0.0
    sigma_n = 1e-40

    ##############################################################################
    # Find the max phase for the SHM.  This corresponds to a stream with zero velocity

    tc_SHM = ckcc.tc(np.zeros(3))
    print "\nThe SHM maximum phase occurs at %f days." % (tc_SHM)
    Er = np.linspace(0.1,10.0,100)
    dR = ckcc.dRdErSHM(Er, tc_SHM, AGe, mDM,sigma_n)

    # Quenching factor?
    #Er = QF^-1 Eee
    #dRdEee=dRdEr*(dEr/dEee)
    #Eee = Er/5
    Eee = ckcc.quench_keVr_to_keVee(Er)
    #Eee = quench_keVee_to_keVr(Er)

    plt.figure()
    #plt.plot(Er,dR)
    plt.plot(Eee,dR)
    plt.xlabel('Eee')
    #plt.show()

    output = "\nSpectrum for the SHM at t=%f\n" % (tc_SHM)
    for x,y in zip(Er,dR):
        print x,y

    plt.figure()
    ############################################################################
    # Make some plots
    ############################################################################
    npts = 100
    xvals = np.linspace(0.1,10.0,npts)
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
        #tot += integrate.quad(dRdErSHM,lo,hi,args=(tc_SHM+day, AGe, mDM,sigma_n))[0]
        #yvals = dRdErSHM(Er, tc_SHM+day, AGe, mDM, sigma_n)
        yvals = ckcc.dRdErSHM(Er,day,AGe,mDM,sigma_n)
        norm_tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        if (j+1)%30==0:

            #yvals = dRdErSHM(Er, tc_SHM+day, AGe, mDM, sigma_n)
            yvals = ckcc.dRdErSHM(Er,day,AGe,mDM,sigma_n)
            Eee = ckcc.quench_keVr_to_keVee(xvals)
            plt.plot(Eee,yvals)

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

    ckcc.vE_t(vEWinter,tc_SHM+365./2)
    vMaxMod = ckcc.normalize(vEWinter)

    print "\n\n\n\nIn galactic coordinates, the stream with maximum modulation:\nDirection:"
    print vMaxMod
    tc_Max=ckcc.tc(vMaxMod)
    print "\nMaximum phase at t=%f.\n" % (tc_Max)

    ############################################################################
    #Now Choose a maximum modulating stream that has a cut-off at the given energy
    ############################################################################
    Er1=3 
    v01=10 #v01 is the dispersion of this stream
    vstr1 = ckcc.vstr(vMaxMod,AGe,tc_SHM,mDM,Er1)
    print "\nStream characteristics for a target with atomic number %.2f and energy cut-off at %f keV:" % (AGe,Er1)
    vstr1 = ckcc.vstr(vMaxMod,AGe,153,mDM,Er1)
    vstr1Vec = np.array([vstr1*vMaxMod[0],vstr1*vMaxMod[1],vstr1*vMaxMod[2]])

    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vstr1,v01)

    print "\nIn earth's frame,";  

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (ckcc.EcutOff(vstr1Vec, tc_Max,AGe, mDM),ckcc.vstrEarth(vstr1Vec,tc_Max))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (ckcc.EcutOff(vstr1Vec, tc_Max+365./2.,AGe, mDM),ckcc.vstrEarth(vstr1Vec,tc_Max+365./2.))

    output = "Spectrum for this stream at t=%f\n" % (tc_Max)
    for i in range(0,11):
        Er=2.5+i*0.1
        output += "%.2E %.8E\n" % (Er,ckcc.dRdErStream(Er, tc_Max, AGe, vstr1Vec, 10,mDM,sigma_n))
    print output



    #The Sagitarius stream may intersect the solar system
    vSag=300 
    v0Sag=10
    vSagHat = np.array([0,0.233,-0.970])
    vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
    tc_Sag = ckcc.tc(vSagVec)

    print "\n\n\n\nThe Sagitarius Stream has a max. phase at %f.\n" % (tc_Sag)


    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vSag,v0Sag) 

    print "\nIn earth's frame,"

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (ckcc.EcutOff(vSagVec, tc_Sag,AGe, mDM),ckcc.vstrEarth(vSagVec,tc_Sag))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (ckcc.EcutOff(vSagVec, tc_Sag+365./2.,AGe, mDM),ckcc.vstrEarth(vSagVec,tc_Sag+365./2.))


    output = "Spectrum for this stream at t=%f\n" % (tc_Sag)
    for i in range(0,11):
        Er=1+i*0.1
        output += "%.2E %.8E\n" % (Er,ckcc.dRdErStream(Er, ckcc.t1, AGe, vSagVec, v0Sag,mDM,sigma_n))
    print output

    vDeb1 = 340
    print "\n\n\n\nDebris spectrum for %.1f\n" % (vDeb1)


    output = "\nSpectrum for debris at t=%f\n" % (tc_SHM)
    for i in range(0,11):
        Er = 0.5+i*0.5
        output += "%.2E %.8E\n" % (Er,ckcc.dRdErDebris(Er, tc_SHM, AGe, mDM, vDeb1,sigma_n))
    print output

    ############################################################################
    # Make some plots
    ############################################################################
    plt.figure()
    npts = 100
    xvals = np.linspace(0.1,10.0,npts)
    yvals = np.zeros(npts)
    for j in range(0,12):
        day = 30*j
        for i,Er in enumerate(xvals):
            #yvals[i] = dRdErDebris(Er, tc_SHM+day, AGe, mDM, vDeb1,sigma_n)
            #yvals[i] = dRdErDebris(Er,day,AGe,mDM,vDeb1,sigma_n)
            #yvals[i] = dRdErStream(Er,day,AGe,vstr1Vec,10,mDM,sigma_n)
            yvals[i] = ckcc.dRdErStream(Er,day,AGe,vSagVec,100,mDM,sigma_n)

        #print yvals
        #print xvals
        #print integrate.trapz(yvals[1:-1],x=xvals[1:-1])
        #print integrate.simps(yvals[1:-1],x=xvals[1:-1])
        #print integrate.quad(dRdErDebris,xvals[1],xvals[-1],args=(tc_SHM+day, AGe, mDM, vDeb1,sigma_n))
        Eee = ckcc.quench_keVr_to_keVee(xvals)
        plt.plot(Eee,yvals)
        #plt.xlabel('Debris, Lisanti')
        plt.xlabel('Stream')


    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()
