
import numpy as np
import random
import math as math
import os #used to change directories os.chdir('directoryName')
import sys #allows for input in the batch file

import csv
import time
# import cprofile  Ivette putin, but doesn't use anyw3ay

#INITIALIZE VARIABLES
random.seed(None) #if a=none then it uses the time. use the same seed for repeatable results
res = 5e-13  #res is the resolution of the simulation. the electron will evaluate whether is scatters or not after travelling this distance in centimeters
#elec #vestigal variable made for number of electrons replaced by main function
extraelecount = 0 #counts the new electrons made
extraelecountrub = 0
extraelecountbuf = 0
ded = 0 #number of dead electrons
spd = 0 #spd is the speed of electrons entering the chamber in the z direction
l = 2.8 #length of chamber in cm. The radius is 1 cm

x = 0 #x,y, and z are the spatial location of the electron in the chamber in centimeters
y = 0
z = 0
vx = 0 #vx,y,z are the components of a vector of the electron's velocity
vy = 0
vz = 0
pol = 0 #polarization of electron, 1 spin up  -1 spin down
energy = 0 #energy of electron
#ev = 0 #initial energy in ev. replaced by main function
d = 0 #d is the distance in meters the electron has travelled since it last scattered
t=0 # t is the   time elapsed for a given electron in seconds

TWOPI = 2*math.pi


j = 0 #j is the number of electrons who have escaped THE GAUNTLET
jpolplus = 0 #number of polarized electrons that exit. The two types of polarization are denoted as +1 or -1
jpolminus = 0
scatternum = 0 #the random number that is generated each scatter to check against the probability
scattertype = 0 #determines what kind of scattering it is

#saves initial values for new electrons created
xsave = []
ysave = []
zsave = []
vxsave = []
vysave = []
vzsave = []
polsave = [] #saves polarization state of new electron
originsave = [] #saves what kind of particle new electron came from. 0=filament 1=rb 2=buffer

#for saving data to file
numberout = []
polout = []  # start with desksL SoI attmom vgongrtrjr.
energyin = []
numincident = []
numrub = []
numbuf = []

#varialbes to determine cross section and probability
lambdavalue = 0
sigmar1 = 0
sigmar2 = 0 # so propabiolity of b2 irrelevant
sigmab1 = 0 # Sigma b means cross section of buffer gas.
sigmab2 = 0
sigmab3 = 0


E= 1.25 #electric field in V/cm
G = 250 #Magnetic Field in Gauss in the +z direction
B = G/10000 #Magnetic Field in Tesla, divides by 10,000 to get telsa
extraelec = 0 #counts how many extra electrons are in the queue. becomes nonzero when a new electron is created


newpol = 0 #for changing pol Unlike pol, this is 0 to 1


origin = 0 #this helps figure out where an electron came from. 0=filament 1=rb 2=buffer


jincident = 0
jrub = 0
jbuf = 0


#the main function that runs the simulation. use it multiple times to test different ev or nXXX values. I've found that using the cluster's 99 hour time limit, it is safe to stay below 12,000 electrons maximum
#bufType is a string, either 'Nitrogen' or 'Ethene'
def testfunction(elec,ev,BGP,nRub,rbPol,bufType):
    nBuf=BGP*3.5e13 # From BSA 4th ed. pg. 94   MAR Assumes T=0 C.  Huh

    #define functions INSIDE my own function!
    def comp2spd(vxIn,vyIn,vzIn):
        speed = math.sqrt(math.pow(vx,2)+math.pow(vy,2)+math.pow(vz,2))
        return speed

    def spd2ev(speedIn):
        #energyfromspeed = ((9.10938356e-31*math.pow((spd/100),2))/3.20435324e-19)
        #Bellis edits
        energyfromspeed = ((9.10938356e-31*math.pow((speedIn/100),2))/3.20435324e-19)
        return energyfromspeed

    def ev2spd(evIn):
        speedfromenergy = math.sqrt((3.20435324e-19*evIn)/9.10938356e-31)
        return speedfromenergy

    #set all of the simulation variables to zero that keep track of
    extraelec = 0
    j = 0
    jpolplus = 0
    jpolminus = 0
    jincident = 0
    jrub = 0
    jbuf = 0
    extraelecountrub = 0
    extraelecountbuf = 0
    ded = 0

    #initialize lists that save some info for output files
    zstartlist = []
    rstartlist = []
    phistartlist = []
    zpollist= [] #where (or if) the electron spin exchanged with rubidium
    energyoutlist = []
    poloutlist = []
    originlist = []
    icount = 0
    while elec > 0:
        elec -= 1
        icount += 1

        if icount%10==0:
            print(f"iteration #{icount}      electron #{elec}")

        if extraelec != 0:  #creates an electron when it was born of ionization
            extraelec -= 1
            #turns all variables into ones from "saved"  electron
            x = xsave[0]
            y = ysave[0]
            z = zsave[0]
            vx = vxsave[0]
            vy = vysave[0]
            vz = vzsave[0]
            pol = polsave[0]
            origin = originsave[0]

            #deletes components from saved electron being used here. the rest of the saved electrons are still stored in the array
            del xsave[0]
            del ysave[0]
            del zsave[0]
            del vxsave[0]
            del vysave[0]
            del vzsave[0]
            del polsave[0]
            del originsave[0]
        else:
            #creates an electron when it was born of filament
            origin = 0

            #in this version the electrons come in more like if they came from a circular filament through a b field. apparently
            radius_i=random.uniform(.03,.05)      # Uncommented this section  MAR version 2 8/15/24
            angle_i=random.uniform(0,(TWOPI))
            x = radius_i*math.cos(angle_i)
            y = radius_i*math.sin(angle_i)
            z = 0
            speed_i = 100 * ev2spd(ev)
            phi_i = 13.5 * ((math.pi) / 180)
            theta_i = angle_i + ((math.pi)/4)
            vx = speed_i * math.sin(phi_i) * math.cos(theta_i)
            vy = speed_i * math.sin(phi_i) * math.sin(theta_i)
            vz = speed_i * math.cos(phi_i)
            pol = random.randrange(-1,2,2)

        #variables that are kept for records
        zstart = z
        rstart = math.sqrt(math.pow(x,2)+math.pow(y,2))
        phistart = math.atan2(x,y)
        zpol = 360 #arbitrary variable to show that the electron did not in fact polarize

        #important variables
        spd = comp2spd(vx,vy,vz)
        energy = spd2ev(spd)
        d = 0
        t=0
        scatternum = random.uniform(0,1) #give a value ot the variable that decides how long the electron flies before scattering
        #end electron creation
        ion = False

        #This is the loop of each electron's joruney
        while True:
            if z < 0: #check if it is behind the chamber
                ded += 1
                break
            elif z > l: #check if it is in front of the chamber
                if math.sqrt(math.pow(x,2) + math.pow(y,2)) > .1: #check if hit the far wall
                    ded += 1
                    break
                else:
                    j += 1 #congrations the electron escaped. records polarization information and other information to be recorded
                    if pol == 1:
                        jpolplus += 1
                    elif pol == -1:
                        jpolminus += 1
                    else:
                        print("Something screwed up involving polarization.")
                    if origin == 0:
                        jincident += 1
                        #print('E_in')
                        print(origin,energy, pol)
                    elif origin == 1:
                        jrub += 1
                        #print('Rb!)')
                        print(origin, energy, pol)
                    elif origin == 2:
                        jbuf += 1
                        #print('created!')
                        print(origin, energy, pol)
                    else:
                        print("Something screwed up involving origin.")
                    zstartlist.append(zstart)
                    rstartlist.append(rstart)
                    phistartlist.append(phistart)
                    zpollist.append(zpol)
                    originlist.append(origin)
                    energyoutlist.append(energy)
                    poloutlist.append(pol)
                    break
            elif math.sqrt(math.pow(x,2) + math.pow(y,2)) > 1: #check if it hit the side walls
                ded += 1
                break
            x += (vx * res) #goes in the direction it moved. this is before checking for scattering because it can't scatter if it doesn't move
            y += (vy * res)
            z += (vz * res)
            t += res
            d += math.sqrt(math.pow(vx * res,2)+math.pow(vy * res,2)+math.pow(vz * res,2))


            #DEFINE CROSS SECTIONS FOR NITROGEN BUFFER GAS
            if bufType == 'Nitrogen':
                if energy < 0.07:
                    sigmar1 = 1.5e-13
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.14:
                    sigmar1 = 5e-14
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.4:
                    sigmar1 = 1.25e-14
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 1.7:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 3.3:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 5e-16
                    sigmab3 = 0
                elif energy < 5:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 8.5:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = 9e-16
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 13:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = 9e-16
                    sigmab2 = (((1.8e-16-1.1e-17)/(13-8.5))*(energy-8.5))+1.1e-17
                    sigmab3 = 0
                elif energy < 15:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = 9e-16
                    sigmab2 = (((1.3e-17-1.8e-16)/(65-13))*(energy-13))+1.8e-16
                    sigmab3 = 0
                elif energy < 65:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = 9.0249e-15*math.pow(energy,-.85117)
                    sigmab2 = (((1.3e-17-1.8e-16)/(65-13))*(energy-13))+1.8e-16
                    sigmab3 = (6.7603E-22*math.pow(energy,3)) - (1.6261E-19*math.pow(energy,2)) + (1.3108E-17*energy)- 1.6229E-16
                else:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = 9.0249e-15*math.pow(energy,-.85117)
                    sigmab2 = 0
                    sigmab3 = (6.7603E-22*math.pow(energy,3)) - (1.6261E-19*math.pow(energy,2)) + (1.3108E-17*energy)- 1.6229E-16

            #DEFINE CROSS SECTIONS FOR ETHENE BUFFER GAS
            if bufType == 'Ethene':
                if energy == 0:
                    sigmar1 = 1.5e-13
                    sigmar2 = 0
                    sigmab1 = 0
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.07:
                    sigmar1 = 1.5e-13
                    sigmar2 = 0
                    sigmab1 = 0
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.14:
                    sigmar1 = 5e-14
                    sigmar2 = 0
                    sigmab1 = 0
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.4:
                    sigmar1 = 1.25e-14
                    sigmar2 = 0
                    sigmab1 = 0
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 0.5:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = 0
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 1.7:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = ((2.725e-15-10e-16)/math.log(2/.75))*math.log(energy)+(2.725e-15-((2.725e-15-10e-16)/math.log(2/.75))*math.log(2))
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 2:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = ((2.725e-15-10e-16)/math.log(2/.75))*math.log(energy)+(2.725e-15-((2.725e-15-10e-16)/math.log(2/.75))*math.log(2))
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 3:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = ((1.75e-15-2.725e-15)/math.log(3/2))*math.log(energy)+(1.75e-15-((1.75e-15-2.725e-15)/math.log(3/2))*math.log(3))
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 5:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 0
                    sigmab1 = ((2.825e-15-1.75e-15)/math.log(8/3))*math.log(energy)+(2.825e-15-((2.825e-15-1.75e-15)/math.log(8/3))*math.log(8))
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 8:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = ((2.825e-15-1.75e-15)/math.log(8/3))*math.log(energy)+(2.825e-15-((2.825e-15-1.75e-15)/math.log(8/3))*math.log(8))
                    sigmab2 = 0
                    sigmab3 = 0
                elif energy < 10.5:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = ((7e-16-2.825e-15)/math.log(100/8))*math.log(energy)+(7e-16-((7e-16-2.825e-15)/math.log(100/8))*math.log(100))
                    sigmab2 = 0
                    sigmab3 = 0
                else:
                    sigmar1 = (1.25e-14)/(energy+0.6)
                    sigmar2 = 6e-16
                    sigmab1 = ((7e-16-2.825e-15)/math.log(100/8))*math.log(energy)+(7e-16-((7e-16-2.825e-15)/math.log(100/8))*math.log(100))
                    sigmab2 = 0
                    sigmab3 = (1e-16)*(3.5e-3)*(math.log(energy/10.5)+(3.9e-1))/(0.0105*(energy/1000)*(1+math.pow(((5.64e-2)/((energy-10.5)/1000)),1.2)))


            lambdavalue = 1/((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf))
            #lambdavalue = 1/((sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf))

            if scatternum >= math.exp(-(d/lambdavalue)): #if the random number is greater than the probability of it not scattering, then it scatters
                d = 0
                scattertype = random.uniform(0,1)

                if scattertype <= ((sigmar1*nRub))/((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf)):

                    #rubidium spin exchange
                  zpol = z
                  newpol = random.uniform(0,1)
                  #rbPol= 100 * math.exp((z-2.8)*.8) #used if rubidium polarization is dependent on z
                  rbPolDec = ((.5*rbPol)+50)/100
                  if newpol <= rbPolDec:
                     pol = 1
                  else:
                     pol = -1

                elif scattertype <= ((sigmar1*nRub)+(sigmar2*nRub))/((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf)):

                    #electron ionizes rubidium
                    if energy <= 7:
                        extraelec += 1
                        elec += 1
                        extraelecountrub += 1

                        #new electron
                        newpol = random.uniform(0,1)
                        #rbPol= 100 * math.exp((z-2.8)*.8) #used if rubidium polarization is dependent on z
                        rbPolDec = ((.5*rbPol)+50)/100
                        if newpol <= rbPolDec:
                            polsave.append(1)
                        else:
                            polsave.append(-1)

                        xsave.append(x)
                        ysave.append(y)
                        zsave.append(z)
                        originsave.append(1)
                        theta = random.uniform(0,TWOPI)
                        randphi = random.uniform(0,1)
                        phi = math.acos((2*randphi)-1)
                        vx2 = spd * math.sin(phi) * math.cos(theta)
                        vy2 = spd * math.sin(phi) * math.sin(theta)
                        vz2 = spd * math.cos(phi)
                        vxsave.append(vx2)
                        vysave.append(vy2)
                        vzsave.append(vz2)

                        #old electron
                        spd = 100 * ev2spd(energy)
                        theta = random.uniform(0,TWOPI)
                        randphi = random.uniform(0,1)
                        phi = math.acos((2*randphi)-1)
                        vx = spd * math.sin(phi) * math.cos(theta)
                        vy = spd * math.sin(phi) * math.sin(theta)
                        vz = spd * math.cos(phi)


                    else:
                        extraelec += 1
                        elec += 1
                        extraelecountrub += 1

                        #new electron
                        newpol = random.uniform(0,1)
                        #newpol = random.uniform(0,1)
                        #rbPol= 100 * math.exp((z-2.8)*.8) #used if rubidium polarization is dependent on z
                        rbPolDec = ((.5*rbPol)+50)/100
                        if newpol <= rbPolDec:
                            polsave.append(1)
                        else:
                            polsave.append(-1)

                        xsave.append(x)
                        ysave.append(y)
                        zsave.append(z)
                        originsave.append(1)
                        spd2 = 100 * ev2spd(1)
                        theta = random.uniform(0,TWOPI)
                        randphi = random.uniform(0,1)
                        phi = math.acos((2*randphi)-1)
                        vx2 = spd2 * math.sin(phi) * math.cos(theta)
                        vy2 = spd2 * math.sin(phi) * math.sin(theta)
                        vz2 = spd2 * math.cos(phi)
                        vxsave.append(vx2)
                        vysave.append(vy2)
                        vzsave.append(vz2)

                        #old electron
                        spd2 = 100 * ev2spd((energy-6))
                        vx = vx * (spd2 / spd)
                        vy = vy * (spd2 / spd)
                        vz = vz * (spd2 / spd)

                elif scattertype <= ((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf))/((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf)):
                #if scattertype <= ((sigmab1*nBuf))/((sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf)):

                    #elastic scattering from nitrogen
                    theta = random.uniform(0,TWOPI)
                    randphi = random.uniform(0,1)
                    phi = math.acos((2*randphi)-1)
                    vx = spd * math.sin(phi) * math.cos(theta)
                    vy = spd * math.sin(phi) * math.sin(theta)
                    vz = spd * math.cos(phi)

                elif scattertype <= ((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf))/((sigmar1*nRub)+(sigmar2*nRub)+(sigmab1*nBuf)+(sigmab2*nBuf)+(sigmab3*nBuf)):

                    #energy loss from nitrogen
                    if energy > 16:
                        ion = True
                    else:
                        vx = 0
                        vy = 0
                        vz = 0

                else:

                    #electron ionizes buffer gas
                    if bufType == 'Nitrogen':

                        if energy <= 25:
                            extraelec += 1
                            elec += 1
                            extraelecountbuf += 1

                            #new electron pt 1
                            newpol = random.uniform(0,1)
                            if newpol <= 0.5:
                                polsave.append(1)
                            else:
                                polsave.append(-1)
                            xsave.append(x)
                            ysave.append(y)
                            zsave.append(z)
                            originsave.append(2)
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx2 = spd * math.sin(phi) * math.cos(theta)
                            vy2 = spd * math.sin(phi) * math.sin(theta)
                            vz2 = spd * math.cos(phi)
                            vxsave.append(vx2)
                            vysave.append(vy2)
                            vzsave.append(vz2)

                            #old electron
                            spd = 100 * ev2spd(((energy-15)/2))
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx = spd * math.sin(phi) * math.cos(theta)
                            vy = spd * math.sin(phi) * math.sin(theta)
                            vz = spd * math.cos(phi)

                        else:
                            extraelec += 1
                            elec += 1
                            extraelecountbuf += 1
                            #new electron
                            newpol = random.uniform(0,1)
                            if newpol <= 0.5:
                                polsave.append(1)
                            else:
                                polsave.append(-1)
                            xsave.append(x)
                            ysave.append(y)
                            zsave.append(z)
                            originsave.append(2)
                            spd2 = 100 * ev2spd(5)
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx2 = spd2 * math.sin(phi) * math.cos(theta)
                            vy2 = spd2 * math.sin(phi) * math.sin(theta)
                            vz2 = spd2 * math.cos(phi)
                            vxsave.append(vx2)
                            vysave.append(vy2)
                            vzsave.append(vz2)
                            #old electron
                            spd2 = 100 * ev2spd((energy-20))
                            vx = vx * (spd2 / spd)
                            vy = vy * (spd2 / spd)
                            vz = vz * (spd2 / spd)


                    if bufType == 'Ethene': #ethene ionization

                        if energy <= 20.5:
                            extraelec += 1
                            elec += 1
                            extraelecountbuf += 1
                            #new electron
                            newpol = random.uniform(0,1)
                            if newpol <= 0.5:
                                polsave.append(1)
                            else:
                                polsave.append(-1)

                            xsave.append(x)
                            ysave.append(y)
                            zsave.append(z)
                            originsave.append(2)
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx2 = spd * math.sin(phi) * math.cos(theta)
                            vy2 = spd * math.sin(phi) * math.sin(theta)
                            vz2 = spd * math.cos(phi)
                            vxsave.append(vx2)
                            vysave.append(vy2)
                            vzsave.append(vz2)

                            #old electron
                            spd = 100 * ev2spd(((energy-10.5)/2))
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx = spd * math.sin(phi) * math.cos(theta)
                            vy = spd * math.sin(phi) * math.sin(theta)
                            vz = spd * math.cos(phi)

                        else:
                            extraelec += 1
                            elec += 1
                            extraelecountbuf += 1

                            #new electron
                            newpol = random.uniform(0,1)
                            if newpol <= 0.5:
                               polsave.append(1)
                            else:
                                polsave.append(-1)

                            xsave.append(x)
                            ysave.append(y)
                            zsave.append(z)
                            originsave.append(2)
                            spd2 = 100 * ev2spd(5)
                            theta = random.uniform(0,TWOPI)
                            randphi = random.uniform(0,1)
                            phi = math.acos((2*randphi)-1)
                            vx2 = spd2 * math.sin(phi) * math.cos(theta)
                            vy2 = spd2 * math.sin(phi) * math.sin(theta)
                            vz2 = spd2 * math.cos(phi)
                            vxsave.append(vx2)
                            vysave.append(vy2)
                            vzsave.append(vz2)

                            #old electron
                            spd2 = 100 * ev2spd((energy-15.5))
                            vx = vx * (spd2 / spd)
                            vy = vy * (spd2 / spd)
                            vz = vz * (spd2 / spd)

                scatternum = random.uniform(0,1) #resets scatter number for next collision

            #Change in vx and vy due to the magnetic field
            if vx != 0 or vy != 0:
                vxy1 = math.sqrt(math.pow(vx,2)+math.pow(vy,2))
                deltavx = -1 * 1.758820023e11 * vy * B * res
                deltavy = 1.758820023e11 * vx * B * res
                vx+=deltavx
                vy+=deltavy
                vxy2 = math.sqrt(math.pow(vx,2)+math.pow(vy,2))
                vx = vx * (vxy1/vxy2)
                vy = vy * (vxy1/vxy2)

            #change in vz due to the electric field
            vz += ((1.7587*math.pow(10,15))*E*res)

            spd = comp2spd(vx,vy,vz)
            if ion == True:
                energy = energy - 16
            else:
                energy = spd2ev(spd)
            ion = False

    #print((extraelecountrub+extraelecountbuf),"electrons were created in the chamber.")
    print((extraelecountbuf+extraelecountrub),"electrons were created in the chamber.")
    print(extraelecountrub," rubidium electrons were created in the chamber.")
    print(extraelecountbuf, " ", bufType, " electrons were created in the chamber.")
    print(ded,"electrons have met with a terrible fate.")
    print(j,"electrons have exited THE GAUNTLET!")
    print("And whadda ya know",jpolplus,"electrons were polarized 1 and",jpolminus,"electrons were polarized -1.")
    print()

    # Bellis edits
    final_polarization = 360
    if not (jpolplus==0 and jpolminus==0):
        final_polarization = (jpolplus-jpolminus)/(jpolplus+jpolminus)

    polout.append(final_polarization)
    '''
    if jpolplus != 0 or jpolminus !=0:
        print("Polarization is",(jpolplus-jpolminus)/(jpolplus+jpolminus))
        polout.append((jpolplus-jpolminus)/(jpolplus+jpolminus))
    else:
        polout.append((360)) #arbitrary number chosen to show that there was an error because there was zero electrons polarized in either direction
    '''

    print(f"OUTPUT {extraelecountbuf+extraelecountrub} {extraelecountrub} {extraelecountbuf} {bufType} {ded} {j} {jpolplus} {jpolminus} {(jpolplus-jpolminus)/(jpolplus+jpolminus)}\n")

 #   data = np.array([zstartlist, rstartlist, phistartlist, zpollist, originlist, energyoutlist, poloutlist],dtype='f')
 #   data = data.T

    #name the file whatever you want. It saves some info about the individual electrons that came out.
#    filename = '/content/drive/MyDrive/MCAll.csv'.format(ev, BGP, bufType, rbPol, nRub)
#    np.savetxt(filename,data,fmt=['%.5f','%.5f','%.5f','%.5f','%.0f','%.4f','%.0f'],delimiter=",",header='A=Z Start;B=Radius Start;C=Phi Start;D=Last Polarization (Z);E=Origin (0=Fil;1=Rub;2=Buf);F=Energy Out (eV);G=Polarization')

    #saves information for the data file of the entire run
 #   energyin.append(ev)
 #   numberout.append(j)
 #   numincident.append(jincident)
 #   numrub.append(jrub)
 #   numbuf.append(jbuf)

#switch back to parameters originally in code on 8/16/24, using 2 eV and 50 % Prb instead of 1ev (or specified) and 99% PRb
'''
elec= 200  #int(sys.argv[1])
ev=   100   #float(sys.argv[2])
BGP=   200   #float(sys.argv[3])
nRub=  1.6e13   #float(sys.argv[4])
rbPol= 99.9  #float(sys.argv[5]) This is actually the rubidium polarization, from 100 (assume perc4nt) to -100
bufType= 'Nitrogen' #sys.argv[6]
'''
elec= int(sys.argv[1])
ev=   float(sys.argv[2])
BGP=   float(sys.argv[3])
nRub=  float(sys.argv[4])
rbPol= float(sys.argv[5]) # This is actually the rubidium polarization, from 100 (assume perc4nt) to -100
bufType= sys.argv[6]

print("MAR1MB Num Elec:", elec)
print("Initial electron energy was",ev,"EV.")
print("Buffer gas type was", bufType," so there")
print(bufType, "concentration was","{:.2e}".format(BGP))
print("Rubidium concentration was","{:.2e}".format(nRub))
print("Rubidium pol was","{:.2e}".format(rbPol))

testfunction(elec, ev, BGP, nRub, rbPol, bufType)

print()
print("MAR Num Elec:", elec)
print("Initial electron energy was",ev,"EV.")
print("Buffer gas type was", bufType," so there")
print(bufType, "concentration was","{:.2e}".format(BGP))
print("Rubidium concentration was","{:.2e}".format(nRub))
print("Rubidium pol was","{:.2e}".format(rbPol))

polout = [1]  # changed from [1]
#data = np.array([energyin, polout, numberout, numincident, numrub, numbuf],dtype='f')
#data = data.T

#name this whatever you want. this is the data file about the set of electrons
#filename = '/content/drive/MyDrive/MCum.csv'S.format(ev, BGP, bufType, rbPol, nRub)
#np.savetxt(filename,data,fmt=['%.1f','%.6f','%.0f','%.0f','%.0f','%.0f'],delimiter=",",header='A=Ei;B=Polarization;C=Number Out;D=Number Out (Fil);E=Number Out (Rb);F=Number Out (Buf)')
#commented out 1/13/25

#print('E_in')
#print(energyin)
#print('#_out')
#print(numberout)
#print('#_inc')
#print(numincident)
#print('#_buf')
#print(numbuf)


# This line should be uncommented if you are not running the program
# on the cluster and you want to view the output that was printed
# on the screen before it disappears.

