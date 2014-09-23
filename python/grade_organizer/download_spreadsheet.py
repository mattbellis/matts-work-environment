# coding: utf-8

import gspread
import sys

if len(sys.argv)<2:
    print "Need password!"
    exit()

password = sys.argv[1]

c = gspread.Client(auth=("matthew.bellis@gmail.com", password))
c.login()
# PHYS 110
s = c.open_by_url('https://docs.google.com/spreadsheets/d/1tF2HIqiTYcPOpJsA3pfc_jQB4W7Hxa9nE_OdxNkJq2c/edit#gid=53530124')
w = s.get_worksheet(1)
d = w.get_all_values()
for val in d:
    #print val
    mystring = ",".join(val)
    print mystring
