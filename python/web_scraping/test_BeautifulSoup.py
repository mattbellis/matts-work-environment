from BeautifulSoup import BeautifulSoup 

import urllib2
import re

page = urllib2.urlopen("http://arxiv.org/list/hep-ex/new")
soup = BeautifulSoup(page)

#print soup.prettify()

count = 0
for incident in soup('dl'):

    print " ----------\n%d\n-----------" % (count)
    print incident

    for title in incident('div', attrs={"class":"list-title"}):

        #print title

        n = len(title.contents)
        if n>=3:
            print title.contents[2]

    for abstract in incident('p'):
        print abstract.contents[0]

    count += 1

exit()

#for incident in soup('p'):
for incident in soup('div', attrs={"class":"list-title"}):
    print "-------"
    n = len(incident.contents)
    if n>=3:
        print incident.contents[2]

for incident in soup('p'):
    print incident.contents[0]
