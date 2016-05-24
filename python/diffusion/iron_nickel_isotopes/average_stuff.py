import numpy as np
import sys

infile = open('values.dat')

vals = np.loadtxt(infile,dtype='str')

xtot = vals.transpose()[6].astype('float')
xerrtot = vals.transpose()[8].astype('float')

print xtot
print xerrtot


for i in range(0,2):
    x,xerr = None,None
    if i==0:
        print "Fe:"
        print " ----- "
        x = xtot[0:2]
        xerr = xerrtot[0:2]
    else:
        print "Ni:"
        print " ----- "
        x = xtot[2:]
        xerr = xerrtot[2:]

    for a,b in zip(x,xerr):
        print "%f +/- %f" % (a,b)

    print 
    print "Weighted average:       %f" % ((x*xerr**2).sum()/(xerr**2).sum())
    print "Combined uncertainties: %f" % np.sqrt((xerr**2).sum())
    print

    print "Mean:     %f " % (np.mean(x))
    print "Std. dev: %f " % (np.std(x))

    print "\n"
