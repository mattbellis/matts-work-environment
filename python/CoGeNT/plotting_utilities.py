import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import scipy.integrate as integrate

#from RTMinuit import *

#import chris_kelso_code as dmm
import chris_kelso_code_cython as dmm

################################################################################
# Plot WIMP signal
################################################################################
def plot_wimp_er(x,AGe,mDM,sigma_n,time_range=[1,365],model='shm'):

    if not (model=='shm' or model=='stream' or model=='debris'):
        print "Not correct model for plotting WIMP PDF!"
        print "Model: ",model
        exit(-1)

    # For debris flow. (340 m/s)
    vDeb1 = 340

    # Assume that the energy passed in is in keVee
    keVr = dmm.quench_keVee_to_keVr(x)

    n = 0
    for org_day in range(time_range[0],time_range[1],1):

        day = (org_day+338)%365.0 #- 151

        if model=='shm':
            n += dmm.dRdErSHM(keVr,day,AGe,mDM,sigma_n)
        elif model=='debris':
            n += dmm.dRdErDebris(keVr,day,AGe,mDM,vDeb1,sigma_n)
        elif model=='stream':
            #The Sagitarius stream may intersect the solar system
            vSag=300
            v0Sag=100
            vSagHat = np.array([0,0.233,-0.970])
            vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
            streamVel = vSagVec
            streamVelWidth = v0Sag
            n += dmm.dRdErStream(keVr,day,AGe,streamVel,streamVelWidth,mDM,sigma_n)

    return n

################################################################################

def plot_wimp_day(org_day,AGe,mDM,sigma_n,e_range=[0.5,3.2],model='shm'):

    if not (model=='shm' or model=='stream' or model=='debris'):
        print "Not correct model for plotting WIMP PDF!"
        print "Model: ",model
        exit(-1)

    # For debris flow. (340 m/s)
    vDeb1 = 340

    n = 0
    day = (org_day+338)%365.0 #- 151
    
    if type(day)==np.ndarray:
        
        n = np.zeros(len(day))
        for i,d in enumerate(day):
            
            x = np.linspace(e_range[0],e_range[1],100)
            keVr = dmm.quench_keVee_to_keVr(x)

            if model=='shm':
                n[i] = (dmm.dRdErSHM(keVr,d,AGe,mDM,sigma_n)).sum()
            elif model=='debris':
                n[i] = (dmm.dRdErDebris(keVr,d,AGe,mDM,vDeb1,sigma_n)).sum()
            elif model=='stream':
                #The Sagitarius stream may intersect the solar system
                vSag=300
                v0Sag=100
                vSagHat = np.array([0,0.233,-0.970])
                vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
                streamVel = vSagVec
                streamVelWidth = v0Sag
                n[i] = (dmm.dRdErStream(keVr,d,AGe,streamVel,streamVelWidth,mDM,sigma_n)).sum()
    else:
        for x in np.linspace(e_range[0],e_range[1],100):

            keVr = dmm.quench_keVee_to_keVr(x)

            if model=='shm':
                n += dmm.dRdErSHM(keVr,day,AGe,mDM,sigma_n)
            elif model=='debris':
                n += dmm.dRdErDebris(keVr,day,AGe,mDM,vDeb1,sigma_n)
            elif model=='stream':
                #The Sagitarius stream may intersect the solar system
                vSag=300
                v0Sag=100
                vSagHat = np.array([0,0.233,-0.970])
                vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
                streamVel = vSagVec
                streamVelWidth = v0Sag
                n += dmm.dRdErStream(keVr,day,AGe,streamVel,streamVelWidth,mDM,sigma_n)
                
    return n

################################################################################
# Plot WIMP signal from debris flow, Lisanti
################################################################################
'''
def plot_wimp_debris_er(x,AGe,mDM,sigma_n,time_range=[1,365]):
    n = 0
    keVr = dmm.quench_keVee_to_keVr(x)
    vDeb1 = 340
    for org_day in range(time_range[0],time_range[1],1):
        day = (org_day+338)%365.0 #- 151
        n += dmm.dRdErDebris(keVr,day,AGe,mDM,vDeb1,sigma_n)
    return n

def plot_wimp_debris_day(org_day,AGe,mDM,sigma_n,e_range=[0.5,3.2]):
    n = 0
    day = (org_day+338)%365.0 #- 151
    vDeb1 = 340
    #day = org_day
    #print day
    if type(day)==np.ndarray:
        #print "here!"
        n = np.zeros(len(day))
        for i,d in enumerate(day):
            #print d
            x = np.linspace(e_range[0],e_range[1],100)
            keVr = dmm.quench_keVee_to_keVr(x)
            #n[i] = (dmm.dRdErSHM(keVr,d,AGe,mDM,sigma_n)).sum()
            n[i] = (dmm.dRdErDebris(keVr,d,AGe,mDM,vDeb1,sigma_n)).sum()
            #print len(tot)
    else:
        for x in np.linspace(e_range[0],e_range[1],100):
            keVr = dmm.quench_keVee_to_keVr(x)
            #n += dmm.dRdErSHM(keVr,day,AGe,mDM,sigma_n)
            n += dmm.dRdErDebris(keVr,day,AGe,mDM,vDeb1,sigma_n)
    return n
'''

################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf_from_lambda(func,bin_width=1.0,scale=1.0,efficiency=None,axes=None,fmt='-',linewidth=1,subranges=None):

    y = None
    plot = None
    srxs = None

    if axes==None:
        axes=plt.gca()

    if subranges!=None:

        srxs = []
        tot_srys = []
        for sr in subranges:
            srxs.append(np.linspace(sr[0],sr[1],1000))
            tot_srys.append(np.zeros(1000))

        totnorm = 0.0
        srnorms = []
        y = []
        plot = []
        for srx,sr in zip(srxs,subranges):
            sry = func(srx)

            # Work in the efficiency
            eff = 1.0
            if efficiency!=None:
                eff = efficiency(srx)
            sry *= eff

            norm = integrate.simps(sry,x=srx)
            srnorms.append(norm)
            totnorm += norm

        for tot_sry,norm,srx,sr in zip(tot_srys,srnorms,srxs,subranges):
            norm /= totnorm

            ypts = func(srx)

            # Work in the efficiency
            eff = 1.0
            if efficiency!=None:
                eff = efficiency(srx)
            ypts *= eff

            #print "norm*scale: ",norm*scale
            ytemp,plottemp = plot_pdf(srx,ypts,bin_width=bin_width,scale=norm*scale,fmt=fmt,axes=axes,linewidth=linewidth)
            y.append(ytemp)
            plot.append(plottemp)
            #tot_sry += y


    return y,plot,srxs


################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf(x,ypts,bin_width=1.0,scale=1.0,efficiency=1.0,axes=None,fmt='-',subranges=None,linewidth=1):

    y = None
    plot = None

    if axes==None:
        axes=plt.gca()

    if subranges!= None:
        totnorm = 0.0
        srnorms = []
        y = []
        plot = []
        for srx,sr in zip(srxs,subranges):
            sry = np.ones(len(srx))
            norm = integrate.simps(sry,x=srx)
            srnorms.append(norm)
            totnorm += norm

        for tot_sry,norm,srx,sr in zip(tot_srys,srnorms,srxs,subranges):
            sry = np.ones(len(srx))
            norm /= totnorm

            ypts = np.ones(len(srx))
            ytemp,plottemp = plot_pdf(srx,ypts,bin_width=bin_width,scale=norm*scale,fmt=fmt,axes=axes)
            y.append(ytemp)
            plot.append(plottemp)
            #tot_sry += y


    else:
        y = np.array(ypts)
        y *= efficiency

        # Normalize to 1.0
        normalization = integrate.simps(y,x=x)
        y /= normalization

        #print "exp int: ",integrate.simps(y,x=x)
        #y *= (scale*bin_width)*efficiency
        y *= (scale*bin_width)

        plot = axes.plot(x,y,fmt,linewidth=linewidth)
        #ytot += y
        #ax0.plot(x,ytot,'b',linewidth=3)

    return y,plot


