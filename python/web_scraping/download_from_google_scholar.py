import urllib2
import sys
import errno

################################################################################
# Download
# Copy the contents of a file from a given URL to a local file.
################################################################################
def download(url):

    print url

    ############################################################################
    # This might be illegal, according to Google's TOS.
    # http://mail.python.org/pipermail/python-list/2012-October/632219.html
    # http://mail.python.org/pipermail/python-list/2012-October/632220.html
    ############################################################################
    hdr = {"User-Agent":"Mozilla/5.0 Cheater/1.0"}
    req = urllib2.Request(url,headers=hdr)

    try:
            page = urllib2.urlopen(req)
            print page.read()
    except urllib2.HTTPError, e:
            print e.fp.read()


################################################################################

if __name__ == '__main__':

    #search_patterns = ['Allan','Weatherwax']
    search_patterns = ['Allan','Weatherwax','Siena']

    url_pattern = ""
    for sp in search_patterns:
        url_pattern += sp
        if sp != search_patterns[-1]:
            url_pattern += "+"
        

    url0 = 'http://scholar.google.com/scholar?hl=en&q='
    url1 = '&btnG=&as_sdt=1%2C33&as_sdtp='
    url = "%s%s%s" % (url0,url_pattern,url1)
    download(url)

