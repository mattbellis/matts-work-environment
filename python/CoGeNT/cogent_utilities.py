import numpy as np
from fitting_utilities import sigmoid

import minuit

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


