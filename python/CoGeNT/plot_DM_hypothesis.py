import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import chris_kelso_code as dmm

import argparse

################################################################################
# Main
################################################################################
def main():
    i = 0.0
    Er = 0.0
    sigma_n = 1e-40

    ############################################################################
    # Parse the command-line arguments.
    ############################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', type=str,\
            default='shm', help='Which DM model (shm (default),stream,debris)')
    parser.add_argument('--target', dest='target', type=str,\
            default='Ge', help='Which target atom (Ge (default),Xe,Na)')
    parser.add_argument('--mDM', dest='mDM', type=float,\
            default=7.0, help='Mass of DM WIMP (default=7.0 GeV)')
    parser.add_argument('--xsec', dest='xsec', type=float,\
            default=1e-40, help='WIMP-proton cross-section (default=1e-40)')

    args = parser.parse_args()

    ############################################################################
    # Grab the necessary information from the command-line parameters.
    ############################################################################
    target_atom = dmm.AGe
    if args.target=='Ge':
        target_atom = dmm.AGe
    elif args.target=='Xe':
        target_atom = dmm.AXe
    elif args.target=='Na':
        target_atom = dmm.ANa

    mDM = args.mDM
    sigma_n = args.xsec

    ##############################################################################
    # Find the max phase for the SHM.  This corresponds to a stream with zero velocity
    ##############################################################################
    Er = None
    Eee = None
    dRdEr = None
    dRdEee = None
    if args.model=='shm':
        tc_SHM = dmm.tc(np.zeros(3))
        print "\nThe SHM maximum phase occurs at %f days." % (tc_SHM)

        # Recoil energy range.
        Er = np.linspace(0.1,10.0,100)
        dRdEr = dmm.dRdErSHM(Er,tc_SHM,target_atom,mDM,sigma_n)

        # Quenching factor to keVee (equivalent energy)
        Eee = dmm.quench_keVr_to_keVee(Er)
        dRdEee = dmm.dRdErSHM(Eee,tc_SHM,target_atom,mDM,sigma_n)

    fig0=plt.figure(figsize=(15,6))
    ax0=fig0.add_subplot(1,3,1)
    ax1=fig0.add_subplot(1,3,2)
    ax2=fig0.add_subplot(1,3,3)

    ax0.plot(Er,dRdEr)
    ax0.set_xlabel('Er')
    ax1.plot(Eee,dRdEee)
    ax1.set_xlabel('Eee')
    plt.show()

    exit()

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
        #tot += integrate.quad(dRdErSHM,lo,hi,args=(tc_SHM+day, target_atom, mDM,sigma_n))[0]
        #yvals = dRdErSHM(Er, tc_SHM+day, target_atom, mDM, sigma_n)
        yvals = dmm.dRdErSHM(Er,day,target_atom,mDM,sigma_n)
        norm_tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        tot += integrate.simps(yvals[1:-1],x=xvals[1:-1])
        if (j+1)%30==0:

            #yvals = dRdErSHM(Er, tc_SHM+day, target_atom, mDM, sigma_n)
            yvals = dmm.dRdErSHM(Er,day,target_atom,mDM,sigma_n)
            Eee = dmm.quench_keVr_to_keVee(xvals)
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

    dmm.vE_t(vEWinter,tc_SHM+365./2)
    vMaxMod = dmm.normalize(vEWinter)

    print "\n\n\n\nIn galactic coordinates, the stream with maximum modulation:\nDirection:"
    print vMaxMod
    tc_Max=dmm.tc(vMaxMod)
    print "\nMaximum phase at t=%f.\n" % (tc_Max)

    ############################################################################
    #Now Choose a maximum modulating stream that has a cut-off at the given energy
    ############################################################################
    Er1=3 
    v01=10 #v01 is the dispersion of this stream
    vstr1 = dmm.vstr(vMaxMod,target_atom,tc_SHM,mDM,Er1)
    print "\nStream characteristics for a target with atomic number %.2f and energy cut-off at %f keV:" % (target_atom,Er1)
    vstr1 = dmm.vstr(vMaxMod,target_atom,153,mDM,Er1)
    vstr1Vec = np.array([vstr1*vMaxMod[0],vstr1*vMaxMod[1],vstr1*vMaxMod[2]])

    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vstr1,v01)

    print "\nIn earth's frame,";  

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (dmm.EcutOff(vstr1Vec, tc_Max,target_atom, mDM),dmm.vstrEarth(vstr1Vec,tc_Max))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (dmm.EcutOff(vstr1Vec, tc_Max+365./2.,target_atom, mDM),dmm.vstrEarth(vstr1Vec,tc_Max+365./2.))

    output = "Spectrum for this stream at t=%f\n" % (tc_Max)
    for i in range(0,11):
        Er=2.5+i*0.1
        output += "%.2E %.8E\n" % (Er,dmm.dRdErStream(Er, tc_Max, target_atom, vstr1Vec, 10,mDM,sigma_n))
    print output



    #The Sagitarius stream may intersect the solar system
    vSag=300 
    v0Sag=10
    vSagHat = np.array([0,0.233,-0.970])
    vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
    tc_Sag = dmm.tc(vSagVec)

    print "\n\n\n\nThe Sagitarius Stream has a max. phase at %f.\n" % (tc_Sag)


    print "\nIn galactic coordinates:"
    print "\nSpeed=%f  Dispersion=%f." % (vSag,v0Sag) 

    print "\nIn earth's frame,"

    print "\nmaximum: Ecutoff=%f stream speed=%f" % (dmm.EcutOff(vSagVec, tc_Sag,target_atom, mDM),dmm.vstrEarth(vSagVec,tc_Sag))
    print "\nminimum: Ecutoff=%f stream speed=%f" % (dmm.EcutOff(vSagVec, tc_Sag+365./2.,target_atom, mDM),dmm.vstrEarth(vSagVec,tc_Sag+365./2.))


    output = "Spectrum for this stream at t=%f\n" % (tc_Sag)
    for i in range(0,11):
        Er=1+i*0.1
        output += "%.2E %.8E\n" % (Er,dmm.dRdErStream(Er, dmm.t1, target_atom, vSagVec, v0Sag,mDM,sigma_n))
    print output

    vDeb1 = 340
    print "\n\n\n\nDebris spectrum for %.1f\n" % (vDeb1)


    output = "\nSpectrum for debris at t=%f\n" % (tc_SHM)
    for i in range(0,11):
        Er = 0.5+i*0.5
        output += "%.2E %.8E\n" % (Er,dmm.dRdErDebris(Er, tc_SHM, target_atom, mDM, vDeb1,sigma_n))
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
            #yvals[i] = dmm.dRdErDebris(Er, tc_SHM+day, target_atom, mDM, vDeb1,sigma_n)
            #yvals[i] = dmm.dRdErDebris(Er,day,target_atom,mDM,vDeb1,sigma_n)
            #yvals[i] = dmm.dRdErStream(Er,day,target_atom,vstr1Vec,10,mDM,sigma_n)
            yvals[i] = dmm.dRdErStream(Er,day,target_atom,vSagVec,100,mDM,sigma_n)

        #print yvals
        #print xvals
        #print integrate.trapz(yvals[1:-1],x=xvals[1:-1])
        #print integrate.simps(yvals[1:-1],x=xvals[1:-1])
        #print integrate.quad(dmm.dRdErDebris,xvals[1],xvals[-1],args=(tc_SHM+day, target_atom, mDM, vDeb1,sigma_n))
        Eee = dmm.quench_keVr_to_keVee(xvals)
        plt.plot(Eee,yvals)
        #plt.xlabel('Debris, Lisanti')
        plt.xlabel('Stream')


    plt.show()

################################################################################
################################################################################
if __name__ == "__main__":
    main()
