#!/usr/bin/env python

import urllib2
import sys
import errno

def download(url):
    """Copy the contents of a file from a given URL
    to a local file.
    """
    import urllib
    # http://stackoverflow.com/questions/4605929/retrying-on-connection-reset
    print url
    outfile = url.split('/')[-1]
    while True:
        try:
            filename, headers = urllib.urlretrieve(url,outfile)
            break
        except IOError as e:
            if e.errno != errno.ECONNRESET:
                raise


################################################################################
if __name__ == '__main__':
    list_of_files = open(sys.argv[1])

    for line in list_of_files:
        f = line.strip()
        download(f)

