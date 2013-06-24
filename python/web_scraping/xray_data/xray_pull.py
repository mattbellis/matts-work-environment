from bs4 import BeautifulSoup

import sys

soup = BeautifulSoup(open(sys.argv[1]))

#print(soup.prettify())

print soup.title

for tr in soup.find_all('tr'):
    #print tr
    #print "===================================="
    output = ""
    for td in tr.find_all('td'):
        #print td
        #print "--------------"
        Z,elem = "test","test"
        vals = []
        for i,p in enumerate(td.find_all('p')):
            p = p.text
            #print p.split()
            if i==0 and len(p.split())==2:
                Z = p.split()[0]
                elem = p.split()[1]
                output += "%s %s: " % (Z,elem)
            else:
                #print p.text.strip().split()
                #print p.text
                #print p.text.decode('ascii','ignore')
                p = p.replace(u'\xa0', "")
                p = p.replace(u'\xa0 ', "")
                p = p.replace(u'\u2020', "")
                p = p.replace(u'\u2014', "")
                p = p.replace('*b', "")
                p = p.replace('*', "")
                val = p.decode('ascii','ignore')
                #print val
                vals.append(val)

        for v in vals:
            output += "%s " % (v)

    #print "OUTPUT"
    print output
    #print e.attrs
    print
