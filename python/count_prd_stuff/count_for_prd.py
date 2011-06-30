#!/usr/bin/env python

import sys

filename = sys.argv[1]

infile = open(filename,"r")

total_chars = 0
total_latex_chars = 0
for line in infile:
    #print line
    latex_entries = line.split('$')
    print latex_entries
    for l in latex_entries:

        print l

        if len(l)>0 and l[0] is not '%':

            #if l.find('\\')>=0:
            if 0:
                num0 = len(l.split('\\'))
                num1 = len(l.split())
                if num0>num1:
                    total_latex_chars += num0
                else:
                    total_latex_chars += num1

            else:
                for char in l:
                    total_chars += 1

            print total_latex_chars
            print total_chars



print "Total latex: %d" % (total_latex_chars)
print "Total non latex: %d" % (total_chars)
print "Total characters: %d" % (total_chars + total_latex_chars)
        
