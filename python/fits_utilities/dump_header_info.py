from astropy.io import fits

import sys

myfilename = sys.argv[1]

for i in range(2):
    print "---------\nHeader %d\n------------" % (i)
    header = fits.getheader(myfilename,i)

    items = header.items()
    for item in items:
        print item
