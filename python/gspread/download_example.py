# coding: utf-8

import gspread
import sys

if len(sys.argv)<2:
    print "Need password!"
    exit()

password = sys.argv[1]

c = gspread.Client(auth=("matthew.bellis@gmail.com", password))
c.login()
c.open_by_url('https://docs.google.com/spreadsheet/ccc?key=0AqEmDaJ8A2rAdDRIc1d1NUV3VEhuRVI2X1BGQ29aYVE&usp=drive_web#gid=1')
s = c.open_by_url('https://docs.google.com/spreadsheet/ccc?key=0AqEmDaJ8A2rAdDRIc1d1NUV3VEhuRVI2X1BGQ29aYVE&usp=drive_web#gid=1')
s.get_worksheet(0)
w = s.get_worksheet(0)
d = w.get_all_values()
for val in d:
    print val
