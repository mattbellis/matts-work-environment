import requests 
from bs4 import BeautifulSoup
import HTMLParser
from HTMLParser import HTMLParser
import sys

# Grab the content from some website
#r = requests.get('http://espn.go.com/mens-college-basketball/teams')

#print r


# Grab the content from some other website
#r = requests.get('http://espn.go.com/mens-college-basketball/team/roster/_/id/399/albany-great-danes')
r = open(sys.argv[1])


# Try to parse the webpage by looking for the tables.
soup = BeautifulSoup(r)

print "\documentclass{article}"
print "\usepackage{graphicx}"
print "\usepackage{subfig}"

#print "\hsize=6.0in"
#print "\\vsize=7.5in"
print "\hoffset=-1.50in"
print "\setlength{\\textwidth}{7.5in}"
print "\setlength{\\textheight}{9in}"
print "\setlength{\\voffset}{0pt}"
print "\setlength{\\topmargin}{0pt}"
print "\setlength{\headheight}{0pt}"
print "\setlength{\headsep}{0pt}"


print "\\begin{document}"
print "\\begin{figure}"
print "\centering"

tables = soup.find_all('table')

icount = 0
for table in tables:
    rows = table.findAll('td')
    for row in rows:
        img = row.findAll('img')
        a = row.findAll('a')
        if len(img)>0 and img[0]['src'].find('jpg')>=0:
            image = img[0]['src']
            name = a[0].string

            image = image.replace(' ','_')

            #print name,image
            if icount%5==4:
                print "\subfloat[%s]{\includegraphics[width=0.19\\textwidth]{%s}}\\\\" % (name,image)
            else:
                print "\subfloat[%s]{\includegraphics[width=0.19\\textwidth]{%s}}\\hfill" % (name,image)

            icount += 1
            

print "\caption{PHYS 110, Section 10}"
print "\end{figure}"
print "\end{document}"

