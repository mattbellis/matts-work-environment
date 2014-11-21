import sys
import re

infile = open(sys.argv[1])

years = ['Freshman','Sophomore','Junior','Senior']

output = ""
count = 0
new_student = True
for line in infile:
    #print line
    vals = line.split()
    if len(vals)>0:
        if new_student is True:
            output += "%d" % (count)

        if re.search('\d\d\d\d',vals[0]):
            output += ",z%s@students.niu.edu" % (vals[0])
            #print output
            new_student = False

        elif re.search('[a-zA-Z],[a-zA-Z]',vals[0]):
            names = vals[0].split(',')
            output += ",%s,%s" % (names[0],names[1])
            #print output

        elif re.search('\-',line):
            names = line.split('-')[0].strip()
            output += ",%s" % (names)
            #print output

        elif len(vals)>=2:
            output += ",%s" % (line.strip())
            #print output

        elif vals[0] in years:
            output += ",%s" % (vals[0].strip())
            new_student = True
            count += 1
            # Print out the final output
            for i in range(0,50):
                output += ","
            print output

        else:
            1
            #print vals

    if new_student is True:
        output = ""




