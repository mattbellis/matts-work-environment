#!/usr/bin/env python

import sys
import shutil

topurl = "https://coursework.stanford.edu/access/content/"

infile = open(sys.argv[1])

links = []
names = []
save_next_line = False
for line in infile:
    if save_next_line:
        lecture_name = line.strip().replace(' ','_').replace('\'','')
        #print "lecture_name"
        if lecture_name.find("sakai")<0:
            print lecture_name
            names.append(lecture_name)
        save_next_line = False

    if line.find('.pdf')>=0 and line.find('coursework')>=0 and (line.find('title="PDF"')>=0 or line.find('title="Unknown"')>=0) and line.find('gif')<0:
        #print line
        url = line.split('=')[1].split('target')[0].strip().split('"')[1].split('/')[-1]
        #print "url"
        print url
        links.append(url)
        save_next_line = True

for l,n in zip(links,names):
    new_name = "%s.pdf" % (n)
    print "cp %s %s" % (l,new_name)
    shutil.copy(l,new_name)

