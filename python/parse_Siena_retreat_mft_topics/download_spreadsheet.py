# coding: utf-8

import gspread
import sys

if len(sys.argv)<2:
    print "Need password!"
    exit()

password = sys.argv[1]
which_sheet = 0


c = gspread.Client(auth=("mbellis@siena.edu", password))
c.login()
# PHYS 110
s = None
w = None
s = c.open_by_url('https://docs.google.com/a/siena.edu/spreadsheets/d/145CoM7cseORIQ87b1JCbISrePh0ZxxiMWIAT8JFv85g/edit#gid=0')

for i in sys.argv[2:]:
    which_sheet = int(i) 
    w = s.get_worksheet(which_sheet)

    d = w.get_all_values()

    #'''
    title = w.title
    name = "%s.csv" % (title)
    outfile = open(name,'w')
    for val in d:
        #print val
        mystring = ",".join(val)
        print mystring
        outfile.write(mystring+"\n")
    outfile.close()
    #'''
