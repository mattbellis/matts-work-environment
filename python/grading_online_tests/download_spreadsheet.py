# coding: utf-8

import gspread
import sys

if len(sys.argv)<2:
    print "Need password!"

password = sys.argv[1]

c = gspread.Client(auth=("mbellis@siena.edu", password))
c.login()
# PHYS 110
#s = c.open_by_url('https://docs.google.com/a/siena.edu/spreadsheets/d/12yClIw-OoRxXYVbwcwfZkVpmgd4tYgalmtCqHILvI94/edit#gid=482626723')
#s = c.open_by_url('https://docs.google.com/a/siena.edu/spreadsheets/d/1k0kxY6-sfbTm09AxPM9QiKwT23QDJaP7N9sheelhpJw/edit#gid=1180300017')
s = c.open_by_url('https://docs.google.com/a/siena.edu/spreadsheets/d/1g2nqd4l0Fi72tE-W42qB2x8t5j3myU9mFJHOkvUtsTU/edit#gid=0')
w = s.get_worksheet(0)

d = w.get_all_values()

name = "%s.csv" % ("thermal_quest_for_absolute_zero")
outfile = open(name,'w')
for val in d:
    print "HERE: ",val
    mystring = "	".join(val)
    print mystring
    mystring = mystring.replace(u'\xa0', u' ')
    outfile.write(mystring+"\n")
outfile.close()
