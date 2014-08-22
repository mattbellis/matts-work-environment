import requests 
from bs4 import BeautifulSoup
import HTMLParser
from HTMLParser import HTMLParser
import sys

# Open the file.
r = open(sys.argv[1])

# Try to parse the webpage by looking for the tables.
soup = BeautifulSoup(r)

print "\documentclass{article}"
print "\usepackage{graphicx}"
print "\usepackage{subfig}"

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
    
    if table['class'][0]=='datadisplaytable':
       
        rows = table.findAll('tr')
        
        image = None
        name = None
        for row in rows:
            cols = row.findAll('td')
        
            for col in cols:
                
                img = col.findAll('img')
                
                a = col.findAll('p')
                if len(img)>0 and img[0]['src'].find('jpg')>=0:
                    image = img[0]['src']
                    image = image.replace(' ','_')
                if len(a)>0 and a[0]['class']==['leftaligntext']:
                    name = a[0].string


                if name is not None and image is not None:
                    if icount%5==4:
                        print "\subfloat[%s]{\includegraphics[width=0.19\\textwidth]{%s}}\\\\" % (name,image)
                    else:
                        print "\subfloat[%s]{\includegraphics[width=0.19\\textwidth]{%s}}\\hfill" % (name,image)

                    image = None
                    name = None

                    icount += 1
                    

print "\caption{PHYS 110, Section 10}"
print "\end{figure}"
print "\end{document}"

