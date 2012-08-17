import sys
import os
from HTMLParser import HTMLParser
import BeautifulSoup as bs

################################################################################
def find_caption(line):
    
    # Get caption
    cap0 = line.find('<td>',line.find('Caption')) + 4
    cap1 = line.find('</td>',cap0)
    #print cap0,cap1
    caption = line[cap0:cap1]
    return caption
################################################################################

################################################################################
def find_filename(line):
    
    # Get caption
    x0 = line.find('<title>',0) + 7
    x1 = line.find('</title>',x0)
    #print x0,x1
    filename = line[x0:x1]
    return filename
################################################################################

################################################################################
def find_size(line):
    
    # Get caption
    x0 = line.find('height=',0) + 7
    x1 = line.find('>',x0)
    #print x0,x1
    height = line[x0:x1]

    hx0 = x0
    x0 = line.find('width=',hx0-20) + 6
    x1 = line.find(' height',x0)
    #print x0,x1
    width = line[x0:x1]

    return width,height

################################################################################

dir_name = sys.argv[1]
frametitle = "XXX"

if len(sys.argv)>2:
    frametitle = sys.argv[2]

if os.access( dir_name, os.W_OK ):

    files = os.listdir(dir_name)
    #print files
    files.sort()
    #print files

    for f in files:
        count = 0
        if 'html' in f:

            #print '---------'
            filename = "%s/%s" % (dir_name,f)
            hfile = open(filename)
            for line in hfile:
                caption = find_caption(line)
                filename = find_filename(line)
                width,height = find_size(line)
                imagename = filename.split('.')[0]

                imagename = imagename.replace('_','\_')

                caption = caption.replace('&#039;',"'")
                caption = caption.replace('&asymp;',"$\\approx$")

                #print imagename
                #print filename
                #print caption
                #print width,height

                print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                print "\\begin{frame}[T]"
                print "\\frametitle{%s}" % (frametitle)
                print ""
                print "    \\begin{figure}[H]"
                print "    \\includegraphics[width=%s\\textwidth]{figures/%s}" % (float(width)/1000.0,filename)
                print "    \\label{lab%d}" % (count)
                print "    \\caption{%s. %s}" % (imagename,caption)
                print "    \\end{figure}"
                print ""
                print "\\end{frame}"
                print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                
                count += 1





else:
    print "Directory does not exist!"
    exit(-1)
