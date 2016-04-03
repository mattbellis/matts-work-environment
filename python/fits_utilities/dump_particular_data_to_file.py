from astropy.io import fits

import numpy as np

import sys

myfilename = sys.argv[1]

myfile = fits.open(myfilename)

fits.getheader
scidata = myfile[1].data

z = scidata['Z']
ra = scidata['PLUG_RA']
dec = scidata['PLUG_DEC']
object_class = scidata['CLASS']

z = z[object_class=='GALAXY']
ra = ra[object_class=='GALAXY']
dec = dec[object_class=='GALAXY']

for a,b,c in zip(ra,dec,z):
    print "%f %f %f" % (a,b,c)

#np.savetxt('test.out', np.transpose([ra,dec,z]),fmt='%f')

