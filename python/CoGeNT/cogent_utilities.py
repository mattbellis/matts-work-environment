import numpy as np
#from fitting_utilities import sigmoid
#from cogent_pdfs import sigmoid

from scipy import integrate

from scipy.interpolate import interp1d

import lichen.pdfs as pdfs

#import minuit
import iminuit as minuit

import scipy.signal as signal

import scipy.stats as stats

################################################################################
# Sigmoid function.
################################################################################
def sigmoid(x,thresh,sigma,max_val):

    ret = max_val / (1.0 + np.exp(-(x-thresh)/(thresh*sigma)))

    return ret



################################################################################
# Conversion 0
# Amplitude (V) to energy (keV)
################################################################################
def amp_to_energy(amplitude, calibration=0):

    energy = 0

    if calibration==0:

        #print "Using calibration 0"

        # Used for the low-energy channel
        energy = 63.7*amplitude
        #energy = 63.7*amplitude + 0.013 # From Nicole


    elif calibration==1:

        #print "Using calibration 1"

        # Used for the higher-energy channel
        energy = (63.049*amplitude) + 0.12719

    elif calibration==2:

        #print "Using calibration 2"

        # Used for the higher-energy channel, when studying the K-shell peaks. Gives
        # a better fit to this region, but worse fit at low energies
        energy = (61.909*amplitude) + 0.28328

    elif calibration==999:

        # No calibration. Data is in keVee

        energy = amplitude

    return energy


################################################################################
# CoGeNT trigger efficiency function.
################################################################################
def cogent_efficiency(data,threshold,sigmoid_sigma,max_val):

    indices = np.zeros(len(data[0]),dtype=np.int)
    for i,pt in enumerate(data[0]):
        if np.random.random()<sigmoid(pt,threshold,sigmoid_sigma,max_val):
            indices[i] = 1

    data[0] = data[0][indices==1]
    data[1] = data[1][indices==1]

    return data


################################################################################
# Return energy and day
################################################################################
def get_cogent_data(infile_name,first_event=0.0,calibration=0):

    infile = open(infile_name)
    content = np.array(infile.read().split()).astype('float')

    ndata = len(content)/2

    # Get time
    index = np.arange(0,ndata*2,2)

    tseconds = content[index]
    tdays = (tseconds-first_event)/(24.0*3600.0) + 1.0

    # Get energy
    index = np.arange(1,ndata*2+1,2)

    amplitudes = content[index]
    energies = amp_to_energy(amplitudes,calibration)

    return tdays,energies


################################################################################
# Return energy and day and rise time
################################################################################
def get_3yr_cogent_data(infile_name,first_event=0.0,calibration=0):

    infile = open(infile_name)
    content = np.array(infile.read().split()).astype('float')

    ndata = len(content)/3

    # Get time
    index = np.arange(0,ndata*3,3)

    tseconds = content[index]
    tdays = (tseconds-first_event)/(24.0*3600.0) + 1.0

    # Get energy
    energies = content[index+1]
    rise_time = content[index+2]

    return tdays,energies,rise_time


################################################################################
# Print data
################################################################################
def print_data(energies,tdays):

    output = ""
    i = 0
    for e,t in zip(energies,tdays):
        if e<3.3:
            output += "%7.2f " % (t)
            i+=1
        if i==10:
            print output
            output = ""
            i=0

################################################################################
# Cut events from an arbitrary dataset that fall outside a set of ranges.
################################################################################
def cut_events_outside_range(data,ranges):

    index = np.ones(len(data[0]),dtype=np.int)
    for i,r in enumerate(ranges):
        if len(r)>0:
            index *= ((data[i]>r[0])*(data[i]<r[1]))

    '''
    for x,y in zip(data[0][index!=True],data[1][index!=True]):
        print x,y
    '''

    for i in xrange(len(data)):
        #print data[i][index!=True]
        data[i] = data[i][index==True]

    return data

################################################################################
# Cut events from an arbitrary dataset that fall outside a set of sub-ranges.
################################################################################
def cut_events_outside_subrange(data,subrange,data_index=0):

    index = np.zeros(len(data[data_index]),dtype=np.int)
    for r in subrange:
        #print r[0],r[1]
        index += ((data[data_index]>r[0])*(data[data_index]<r[1]))
        #print data[1][data[1]>107.0]

    #print index[index!=1]
    for i in xrange(len(data)):
        #print data[i][index!=True]
        data[i] = data[i][index==True]

    return data


################################################################################
# Precalculate the probabilities for all the lognormal distributions.
################################################################################
def rise_time_prob(rise_time,energy,mu_k,sigma_k,xlo,xhi):

    # Pull out the constants for the polynomials.
    ma0 = mu_k[0]
    ma1 = mu_k[1]
    ma2 = mu_k[2]

    sa0 = sigma_k[0]
    sa1 = sigma_k[1]
    sa2 = sigma_k[2]

    allmu = ma0 + ma1*energy + ma2*energy*energy
    allsigma = sa0 + sa1*energy + sa2*energy*energy

    #ret = (1.0/(x*sigma*np.sqrt(2*np.pi)))*np.exp(-((np.log(x)-mu)**2)/(2*sigma*sigma))
    ret = np.zeros(len(rise_time))
    for i in xrange(len(ret)):
        #print rise_time[i],allmu[i],allsigma[i]
        ret[i] = pdfs.lognormal(rise_time[i],allmu[i],allsigma[i],xlo,xhi)
        #print "\t",ret[i]

    return ret

################################################################################
# Precalculate the probabilities for the fast lognormal distributions.
################################################################################
def rise_time_prob_fast_exp_dist(rise_time,energy,mu0,sigma0,murel,sigmarel,numrel,xlo,xhi):

    expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]

    # Pull out the constants for the polynomials.
    fast_mean0 = expfunc(mu0,energy)
    fast_sigma0 = expfunc(mu0,energy)
    fast_num0 = np.ones(len(rise_time)).astype('float')

    # The entries for the relationship between the broad and narrow peak.
    fast_mean_rel = expfunc(murel,energy)
    fast_sigma_rel = expfunc(sigmarel,energy)
    fast_logn_num_rel = expfunc(numrel,energy)

    fast_mean1 = fast_mean0 - fast_mean_rel
    fast_sigma1 = fast_sigma0 - fast_sigma_rel
    fast_num1 = fast_num0 / fast_logn_num_rel

    tempnorm = (fast_num0+fast_num1)

    fast_num0 /= tempnorm
    fast_num1 /= tempnorm

    print "Fast NUMS 0 and 1: ",fast_num0[10],fast_num1[10]

    ret = np.zeros(len(rise_time))
    for i in xrange(len(ret)):
        #print rise_time[i],allmu[i],allsigma[i]
        pdf0 = pdfs.lognormal(rise_time[i],fast_mean0[i],fast_sigma0[i],xlo,xhi)
        pdf1 = pdfs.lognormal(rise_time[i],fast_mean1[i],fast_sigma1[i],xlo,xhi)
        ret[i] = fast_num0[i]*pdf0 + fast_num1[i]*pdf1
        #print "\t",ret[i]

    return ret

################################################################################
# Precalculate the probabilities for all the lognormal distributions.
################################################################################
def rise_time_prob_exp_progression(rise_time,energy,mu_k,sigma_k,xlo,xhi):

    expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]

    # Pull out the constants for the polynomials.
    allmu = expfunc(mu_k,energy)
    allsigma = expfunc(sigma_k,energy)

    #ret = (1.0/(x*sigma*np.sqrt(2*np.pi)))*np.exp(-((np.log(x)-mu)**2)/(2*sigma*sigma))
    ret = np.zeros(len(rise_time))
    for i in xrange(len(ret)):
        #print rise_time[i],allmu[i],allsigma[i]
        ret[i] = pdfs.lognormal(rise_time[i],allmu[i],allsigma[i],xlo,xhi)
        #print "\t",ret[i]

    return ret

################################################################################
# Convolve a function with the CoGeNT resolution.
################################################################################
def cogent_convolve(x,y):

    npts = 2
    if type(y)==np.ndarray:
        npts = len(y)

    xpts = np.linspace(-5,5,100)
    #xpts = np.linspace(-5,5,npts)
    #xpts = x
    
    eta = 2.96 # eV
    eta /= 1000.0 # convert to keV
    F = 0.29 # adimensional form factor
    sigman = 69.4 # eV
    sigman /= 1000.0 # convert to keV
    sigman2 = sigman*sigman
    
    '''
    #sigma = np.sqrt(sigman2 + (xpts*eta*F))
    sigma = 0.5
    convolving_pts = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-((xpts-0.0)**2)/(2*sigma*sigma)) # Make this npts as well.
    #convolving_pts = np.exp(-((xpts-0.0)**2)/(2*sigma*sigma)) # Make this npts as well.
    convolving_pts /= convolving_pts.sum()
    '''
    yc = np.zeros(npts)
    #print "x: ",x
    #print "y: ",y

    if type(x)!=np.ndarray:
        x = np.array([x,x+0.0001])

    if type(y)!=np.ndarray:
        y = np.array([y,y+0.0001])

    f = interp1d(x, y)

    #print y

    #exit()

    yc = np.zeros(len(y))

    for i,(xpt,ypt) in enumerate(zip(x,y)):
        #if 1:
        if ypt>0:
            sigma = np.sqrt(sigman2 + (xpt*eta*F)) # sigma is energy dependent
            #sigma = 0.5 + xpt*0.1 # FOR TESTING
            #sigma = 0.1
            #print sigma

            window = 3.0*sigma
            loval = xpt-window
            if loval<min(x):
                loval=min(x)
            hival = xpt+window
            if hival>max(x):
                hival=max(x)

            convolving_term = stats.norm(0.0,sigma)
            xconv = np.linspace(-5,5,5*npts)
            # Use a normalized Gaussian.
            yconv = convolving_term.pdf(xconv)

            ytemp = np.zeros(npts)
            ytemp[i] = y[i]

            if ytemp[i]>0:
                # Convolve a single point in the original function.
                convolved_pdf = signal.fftconvolve(ytemp,yconv,mode='same')

                #print s,x[s],y[s],sigma_conv,convolved_pdf[500]
                # Sum up each of the contributions.
                yc += convolved_pdf


            
            '''
            # THIS IS AN OLD WAY I WAS TRYING TO DO THIS
            print sigma,window,loval,hival
            temp_pts = np.linspace(loval,hival,1000)
            print temp_pts[0],temp_pts[-1]
            
            ytemp = f(temp_pts)

            convolving_term = stats.norm(0.0,sigma)

            if sum(ytemp)>0:
                val = (ytemp*convolving_term.pdf(xpt-temp_pts)).sum()

                yc[i] = val

            '''


    #convolved_function = signal.convolve(y/y.sum(),convolving_pts,mode='same')
    #convolved_function = signal.convolve(y/y.sum(),convolving_pts)
    #convolved_function = signal.fftconvolve(y/y.sum(),convolving_pts,'same')
    #convolved_function = np.convolve(y/y.sum(),convolving_pts,'same')

    #norm = convolved_function.sum()/y.sum()
    norm = integrate.simps(yc,x=x)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it.
    #znpts = len(convolved_function)
    #begin = znpts/2 - npts/2
    #end = znpts/2 + npts/2

    #print "%d %d %d %d" % (npts,znpts,begin,end)

    #return convolved_function[begin:end]/norm,convolving_pts
    #return convolved_function/norm,convolving_pts
    return yc/norm,x



