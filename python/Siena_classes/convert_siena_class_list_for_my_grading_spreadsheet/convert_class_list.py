import numpy as np
import sys

# This works with the .csv downloaded from 
# Siena Class Roster 
# Siena Class Roster with option to download to excel.

infile = open(sys.argv[1])

print "%s,%s,%s,%s,%s,%s,%s,%s" % ("","","","","","","","Quiz HW or Exam")
print "%s,%s,%s,%s,%s,%s,%s,%s" % ("","","","","","","","Assignment")
print "%s,%s,%s,%s,%s,%s,%s,%s" % ("","","","","","","","1/1/2014")
print "%s,%s,%s,%s,%s,%s,%s,%d" % ("Number","Email","Last name","First name","Program","Major","Year",10)

count = 0
for line in infile:
    vals = line.split(",")
    #print vals
    if len(vals)>1 and vals[0] != 'TERM':
        lname = vals[7]
        fname = vals[8]
        year = vals[10]
        program = vals[11]
        major = vals[12]
        email = vals[14]

        print "%d,%s,%s,%s,%s,%s,%d,%s" % (count,email,lname,fname,program,major,int(year)," ")
        count += 1
