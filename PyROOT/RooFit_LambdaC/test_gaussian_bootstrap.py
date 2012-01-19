#!/usr/bin/env python

from math import *
from optparse import OptionParser
import sys
import numpy
import random

from bootstrapping import *

################################################################################
################################################################################
def main(argv):

    ### Command line variables ####
    parser = OptionParser()
    parser.add_option("--mean", dest="mean", default=0.0, help='Mean of gaussian \
            from which to sample')
    parser.add_option("--sigma", dest="sigma", default=0.1, help='Sigma of gaussian \
            from which to sample')
    parser.add_option("--nevents", dest="nevents", default=100, help='Number of\
            trials to run in bootstrapping approach.')
    parser.add_option("--interval", dest="interval", default=0.68, \
            help='Confidence interval over which to calculate the \
            error.')
    parser.add_option("--sample-size", dest="sample_size", default=1000, \
            help='Number of entries to use in bootstrapping sub \
            samples.')
    parser.add_option("--ntrials", dest="ntrials", default=1000, help='Number of\
            trials to run in bootstrapping approach.')

    (options, args) = parser.parse_args()

    ntrials = int(options.ntrials)
    sample_size = int(options.sample_size)
    mean = float(options.mean)
    sigma = float(options.sigma)
    nevents = int(options.nevents)
    interval = float(options.interval)
    
    x = numpy.random.normal(mean,sigma,nevents)

    print x

    m_intervals, m_trials, m = calc_mean_and_subsamples(x,ntrials,sample_size,interval)

    # Calculate the indices over which we might calculate the errors.
    # Interval indices
    n = len(m_trials)
    i_lo = int(n*(0.50 - (interval/2.0)))
    i_hi = int(n*(0.50 + (interval/2.0)))

    print i_lo
    print i_hi

    m_mean = numpy.mean(m_trials)
    m_std =  numpy.std(m_trials)
    # The mean of the intervals over the trials
    m_int_lo =  numpy.mean(m_intervals[0])
    m_int_hi =  numpy.mean(m_intervals[1])
    # The mean of the intervals over the trials
    m1_int_lo =  m_trials[i_lo]
    m1_int_hi =  m_trials[i_hi]

    print "Orginal sample mean:       %6.3f" % (m)
    print "Orginal sample std dev:    %6.3f" % (numpy.std(x))
    print "Bootstrap samples mean:    %6.3f" % (m_mean)
    print "Bootstrap samples std dev: %6.3f" % (m_std)
    print "Bootstrap conf. interval:  %6.3f - %6.3f" % (m_int_lo, m_int_hi)
    print "Bootstrap error estimate:  %6.3f" % ((m_int_hi-m_int_lo)/2.0)
    print "Trials conf. interval:     %6.3f - %6.3f" % (m1_int_lo, m1_int_hi)
    print "Trials error estimate:     %6.3f" % ((m1_int_hi-m1_int_lo)/2.0)
    




################################################################################
################################################
if __name__ == "__main__":
        main(sys.argv)


