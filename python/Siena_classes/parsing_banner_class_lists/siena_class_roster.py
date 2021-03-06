import requests 
from bs4 import BeautifulSoup
import html.parser
from html.parser import HTMLParser
import sys
import os

# Open the file.
r = open(sys.argv[1])

if os.path.exists('./Siena Class Roster_files'):
    os.rename('Siena Class Roster_files','Siena_Class_Roster_files')

# Try to parse the webpage by looking for the tables.
soup = BeautifulSoup(r,"lxml")

print("\documentclass{article}")
print("\\usepackage{graphicx}")
print("\\usepackage{subfig}")
print("\\usepackage{alphalph}")
print("\\renewcommand\\thesubfigure{\\alphalph{\\value{subfigure}}}")

print("\hoffset=-1.50in")
print("\setlength{\\textwidth}{7.5in}")
print("\setlength{\\textheight}{9in}")
print("\setlength{\\voffset}{0pt}")
print("\setlength{\\topmargin}{0pt}")
print("\setlength{\headheight}{0pt}")
print("\setlength{\headsep}{0pt}")
print("\\begin{document}")

maxsubfigs = 20


h2s = soup.find_all('h2')
caption = 'Default'
for h in h2s:
    if h.string.find('Class Roster For')>=0:
        caption = h.string

tables = soup.find_all('table')

icount = 0
closed_figure = False
for table in tables:
    
    if table['class'][0]=='datadisplaytable':
       
        rows = table.findAll('tr')
        
        image = None
        name = None
        for row in rows:
            cols = row.findAll('td')
        
            #print("-------------------------")
            rowcount = -1
            year = "Default"
            major = "Default"
            for col in cols:
                rowcount += 1 

                #print("COLS")
                #print(col)
                
                img = col.findAll('img')
                
                a = col.findAll('p')
                #print("here")
                #print(a)
                if len(img)>0 and img[0]['src'].find('jpg')>=0:
                    image = img[0]['src']
                    image = image.replace(' ','_')
                if len(a)>0 and a[0]['class']==['leftaligntext']:
                    name = a[0].string
                if rowcount == 3:
                    major = a[0].string
                if rowcount == 4:
                    year = a[0].string


                if name is not None and image is not None and rowcount==5:
                    if icount%maxsubfigs==0:
                        #print("\\begin{document}")
                        print("\n")
                        print("\\begin{figure}")
                        print("\centering")
                        closed_figure = False

                    #print(image)
                    #if os.stat(image).st_size < 300:
                    if not os.path.isfile(image):
                        image = './file_not_found.jpg'

                    if icount%5==4:
                        print("\subfloat[%s, %s, %s]{\includegraphics[width=0.15\\textwidth]{%s}}\\\\" % (name,major,year,image))
                        #print("\subfloat[%s]{\includegraphics[width=0.15\\textwidth]{%s}}\\\\" % (name,image))
                    else:
                        print("\subfloat[%s, %s, %s]{\includegraphics[width=0.15\\textwidth]{%s}}\\hfill" % (name,major,year,image))
                        #print("\subfloat[%s]{\includegraphics[width=0.15\\textwidth]{%s}}\\hfill" % (name,image))

                    image = None
                    name = None

                    if icount%maxsubfigs==maxsubfigs-1:
                        print("\caption{%s}" % (caption))
                        print("\end{figure}")
                        closed_figure = True
                        icount = -1

                    icount += 1
                    #print icount
                    

if not closed_figure:
    print("\caption{%s}" % (caption))
    print("\end{figure}")
print("\end{document}")

