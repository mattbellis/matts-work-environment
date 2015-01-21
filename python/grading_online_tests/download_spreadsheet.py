# coding: utf-8

import gspread
import sys

if len(sys.argv)<2:
    print "Need password!"

password = sys.argv[1]

c = gspread.Client(auth=("mbellis@siena.edu", password))
c.login()
# PHYS 110
s = c.open_by_url('https://docs.google.com/a/siena.edu/spreadsheets/d/12yClIw-OoRxXYVbwcwfZkVpmgd4tYgalmtCqHILvI94/edit#gid=482626723')
w = s.get_worksheet(0)

d = w.get_all_values()

name = "%s.csv" % ("thermal_syllabus_quiz")
outfile = open(name,'w')
for val in d:
    #print val
    mystring = "	".join(val)
    print mystring
    outfile.write(mystring+"\n")
outfile.close()
