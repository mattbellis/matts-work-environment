# coding: utf-8

import gspread
import sys

if len(sys.argv)<3:
    print "Need password!"
    exit()

course = sys.argv[1]
password = sys.argv[2]

c = gspread.Client(auth=("matthew.bellis@gmail.com", password))
c.login()
# PHYS 110
s = None
w = None
if course=="phys110":
    s = c.open_by_url('https://docs.google.com/spreadsheets/d/1tF2HIqiTYcPOpJsA3pfc_jQB4W7Hxa9nE_OdxNkJq2c/edit#gid=53530124')
    w = s.get_worksheet(1)
elif course=="phys310":
    s = c.open_by_url('https://docs.google.com/spreadsheets/d/1tF2HIqiTYcPOpJsA3pfc_jQB4W7Hxa9nE_OdxNkJq2c/edit#gid=1083720838')
    w = s.get_worksheet(0)
elif course=="phys260":
    s = c.open_by_url('https://docs.google.com/spreadsheets/d/1vm_xNMUsVUpLOEEIv3XAc52m_gCxFwnJby4uFgO9ZPk/edit#gid=333596407')
    w = s.get_worksheet(0)

d = w.get_all_values()

name = "%s.csv" % (course)
outfile = open(name,'w')
for val in d:
    #print val
    mystring = ",".join(val)
    print mystring
    outfile.write(mystring+"\n")
outfile.close()
