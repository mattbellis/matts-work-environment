import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

import chris_kelso_code as dmm
#import chris_kelso_code_cython as dmm

import plotting_utilities as pu
import lichen.plotting_utilities as plotting_utilities
import fitting_utilities as fu
import cogent_pdfs as cpdf
from cogent_utilities import cogent_convolve

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
    parser.add_argument('--elo', dest='elo', type=float,\
            default=0.0, help='Low energy range for your detector (default=0)')
    parser.add_argument('--ehi', dest='ehi', type=float,\
            default=4.0, help='High energy range for your detector (default=4)')
    parser.add_argument('--cogent', dest='cogent',\
            default=False, action='store_true', help='Use the CoGeNT detector for size and efficiency')

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
    elo = float(args.elo)
    ehi = float(args.ehi)

    ############################################################################
    # Figure
    ############################################################################
    fig0=plt.figure(figsize=(14,4),dpi=100)
    ax0=fig0.add_subplot(1,3,1)
    ax1=fig0.add_subplot(1,3,2)
    ax2=fig0.add_subplot(1,3,3)

    fig0.subplots_adjust(left=0.07, bottom=0.15, right=0.95, wspace=0.2, hspace=None)

    ##############################################################################
    # Find the max phase for the SHM.  This corresponds to a stream with zero velocity
    ##############################################################################
    Er = None
    Eee = None
    dRdEr = None
    dRdEee = None

    # Need this for Sagitarius stream
    vSag=300
    #vSag=500
    v0Sag=25
    #vSag=500
    #v0Sag=10
    vSagHat = np.array([0,0.233,-0.970])
    vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])

    streamVel = vSagVec
    streamVelWidth = v0Sag

    # For debris flow. (340 m/s)
    vDeb1 = 340

    # Efficiency
    efficiency = lambda x: 1.0
    '''
    if args.cogent:
        #eff_scaling = 1.0
        eff_scaling = 0.9 # 3yr data
        max_val = 0.86786
        threshold = 0.345
        sigmoid_sigma = 0.241

        efficiency = lambda x: fu.sigmoid(x,threshold,sigmoid_sigma,max_val)/eff_scaling
    '''


    if args.model=='shm':
        tc_SHM = dmm.tc(np.zeros(3))
        print "\nThe SHM maximum phase occurs at %f days." % (tc_SHM)

    ############################################################################
    # Plot the spectra
    ############################################################################
    for day in [0,60,120,180,240,300]:
        # Recoil energy range.
        Er = np.linspace(0.1,10.0,100)

        if args.model=='shm':
            dRdEr = dmm.dRdErSHM(Er,day,target_atom,mDM,sigma_n)
        elif args.model=='stream':
            dRdEr = dmm.dRdErStream(Er,day,target_atom,streamVel,streamVelWidth,mDM,sigma_n)
        elif args.model=='debris':
            dRdEr = dmm.dRdErDebris(Er,day,target_atom,mDM,vDeb1,sigma_n)

        #wimp(org_day,x,AGe,mDM,sigma_n,efficiency=None,model='shm',vDeb1=340,vSag=300,v0Sag=100):
        y = (day+338)%365.0
        xkeVr = dmm.quench_keVr_to_keVee(Er)
        dRdEr = cpdf.wimp(y,xkeVr,target_atom,mDM,sigma_n,efficiency=None,model='shm')

        leg_title = "day=%d" % (day)
        ax0.plot(Er,dRdEr,label=leg_title)

        # Quenching factor to keVee (equivalent energy)
        Eee = np.linspace(elo,ehi,100)
        # I think we need to do this over a wider range to get the convolution right. 
        #Eee = np.linspace(0.1,ehi,100) 
        Er = dmm.quench_keVee_to_keVr(Eee)

        #func = lambda x: plot_wimp_er(Er,target_atom,mDM,sigma_n,time_range=[day,day+1],model=args.model)

        if args.model=='shm':
            dRdEr = dmm.dRdErSHM(Er,day,target_atom,mDM,sigma_n)
        elif args.model=='stream':
            dRdEr = dmm.dRdErStream(Er,day,target_atom,streamVel,streamVelWidth,mDM,sigma_n)
        elif args.model=='debris':
            dRdEr = dmm.dRdErDebris(Er,day,target_atom,mDM,vDeb1,sigma_n)

        dRdEr *= efficiency(Er)

        #smeared,smeared_x = cogent_convolve(Eee,dRdEr)
        #print smeared-dRdEr

        leg_title = "day=%d" % (day)
        ax1.plot(Eee,dRdEr,label=leg_title)
        #ax1.plot(Eee,smeared,'--',label='smeared')

    ax0.set_xlabel('Er')
    ax0.legend()

    ax1.set_xlabel('Eee')
    ax1.set_xlim(elo,ehi)
    ax1.legend()

    num_wimps = 1.0

    #num_wimps = integrate.dblquad(cpdf.wimp,elo,ehi,lambda x: 1,lambda x:366,args=(target_atom,mDM,sigma_n,efficiency,args.model,vDeb1,vSag,v0Sag),epsabs=0.001)[0]
    num_wimps = 1000.0

    if args.cogent:
        num_wimps *= 0.333

    print "# WIMPs: ",num_wimps

    func = lambda x: pu.plot_wimp_day(x,target_atom,mDM,sigma_n,e_range=[elo,ehi],model=args.model,vSag=vSag,v0Sag=v0Sag,vDeb1=vDeb1)
    plotting_utilities.plot_pdf_from_lambda(func,scale=num_wimps,fmt='k-',linewidth=3,axes=ax2,subranges=[[1,365]])

    ax2.set_ylim(0.0,num_wimps/100.0)

    leg_title = "# WIMPs: %4.1f" % (num_wimps)
    print leg_title
    ax2.legend([leg_title])

    ax2.set_xlabel('Day of year')

    dRdEr = dmm.dRdErSHM(7.0,100.0,target_atom,mDM,sigma_n)

    print "dRdEr: ",dRdEr

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
    #vSagHat = np.array([-0.07247722,  -0.99486114, +0.07069913])
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
