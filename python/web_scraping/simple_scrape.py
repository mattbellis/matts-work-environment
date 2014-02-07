#!/usr/bin/env python

import urllib2
import sys
import errno

################################################################################
# Download
# Copy the contents of a file from a given URL to a local file.
################################################################################
def download(url):

    print url
    outfile = open('temp.txt','w')

    response = urllib2.urlopen(url)
    html = response.read()
    outfile.write(html)




################################################################################

if __name__ == '__main__':

    url = sys.argv[1]
    download(url)

