#!/usr/bin/env python

from math import *
from optparse import OptionParser
import sys
import numpy
import random

################################################################################
################################################################################
def calc_coeff_and_subsamples(x=[1.0],y=[1.0], ntrials=1000, sample_size=1000):

    nentries = len(x)

    # Error check!
    if nentries!=len(y):
        print "calc_coeff_and_subsamples:"
        print "Two lists are not the same length!!!!!!!"
        print
        return -1

    cc = numpy.corrcoef(x,y)[1][0]

    #print sample_size
    #print ntrials

    cc_trials = []
    for n in range(0,ntrials):

        if n%100==0:
            1
            #print n

        # Create the subsamples.
        # There will be sample_size entries in them.
        xtest = []
        ytest = []

        # Create a random list of unique integers. 
        # These will be the events we sample from the passed in samples.
        # This is more like JACKKNIFIING
        #indices = random.sample(xrange(nentries),sample_size)

        # Create a random list that does NOT have to be  unique integers. 
        # These will be the events we sample from the passed in samples.
        # This is more like traditional bootstrapping.
        for i in range(0,sample_size):
            index = random.randint(0,nentries-1)

            xtest.append(x[index])
            ytest.append(y[index])
        
        #print "test vals: %f %f " % (xtest[0],ytest[0])
        # Save the transformed variable.
        # Use arctanh, the arctan hyperbolic function.
        #cc_trials.append(numpy.arctanh(numpy.corrcoef(xtest,ytest)[0][1]))
        # Trying without arctanh
        cc_trials.append(numpy.corrcoef(xtest,ytest)[0][1])

    return cc_trials, cc

################################################################################

################################################################################
def calc_mean_and_subsamples(x=[1.0],ntrials=1000, sample_size=1000, interval=0.50):

    nentries = len(x)

    m = numpy.mean(x)

    #print sample_size
    #print ntrials
    
    # Interval indices
    i_lo = int(sample_size*(0.50 - (interval/2.0)))
    i_hi = int(sample_size*(0.50 + (interval/2.0)))

    print i_lo
    print i_hi

    m_trials = []
    m_intervals = [[],[]]
    for n in range(0,ntrials):

        if n%100==0:
            1
            #print n

        # Create the subsamples.
        # There will be sample_size entries in them.
        xtest = []

        # Create a random list of unique integers. 
        # These will be the events we sample from the passed in samples.
        # This is more like JACKKNIFIING
        #indices = random.sample(xrange(nentries),sample_size)

        # Create a random list that does NOT have to be  unique integers. 
        # These will be the events we sample from the passed in samples.
        # This is more like traditional bootstrapping.
        indices = []
        for i in range(0,sample_size):
            index = random.randint(0,nentries-1)
            xtest.append(x[index])
        
        #print "test vals: %f %f " % (xtest[0],ytest[0])
        m_trials.append(numpy.mean(xtest))

        #print xtest
        mtest = xtest
        mtest.sort()

        m_intervals[0].append(mtest[i_lo])
        m_intervals[1].append(mtest[i_hi])


    return m_intervals, m_trials, m

################################################################################

