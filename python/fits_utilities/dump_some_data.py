from astropy.io import fits

import sys

myfilename = sys.argv[1]

myfile = fits.open(myfilename)

scidata = hdulist[1].data

#z = scidata['Z']

