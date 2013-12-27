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
    outfile = open('pub_record.txt','w')

    response = urllib2.urlopen(url)
    html = response.read()
    print html
    for val in html.split():
        if val.find('export')>=0 and val.find('BibTeX')>=0:
            print val
            new_url = val.split('"')[1]
            print new_url
            #response = urllib2.urlopen(url)
            #html = response.read()
            #print html
    #outfile.write(html)




################################################################################

if __name__ == '__main__':

    url = 'http://inspirehep.net/search?ln=en&ln=en&p=find+author+m.+bellis&action_search=Search&sf=&so=d&rm=&rg=25&sc=0&of=hb'
    download(url)

