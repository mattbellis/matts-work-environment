import numpy as np

import csv
import sys

infilename = sys.argv[1]

students = []

with open(infilename, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i,row in enumerate(spamreader):
        print(len(row))
        if i==0:
            assignmentnames = row[3:]
            fname = row[0]
            lname = row[1]
            emailaddress = row[2]
        elif i==1:
            dates = row[3:]
        elif i==2:
            points = np.array(row[3:]).astype(float)
        elif i>3:
            student = {}
            fname = row[0]
            lname = row[1]
            emailaddress = row[2]
            student["name"] = "%s %s" % (fname, lname)
            student["email"] = emailaddress
            assignments = []
            for a,b,c,d in zip(assignmentnames,dates,points,row[3:]):
                if d!='' and d[0].isdigit():
                    print(a,b,c,d)
                    assignment = {}
                    assignment["name"] = a
                    assignment["date"] = b
                    assignment["maxpoints"] = c
                    assignment["score"] = float(d)
                    assignments.append(assignment)
            student["assignments"] = assignments
            students.append(student)

    #print(points)
print(students)
s = students[0]
for key in s.keys():
    print(s[key])
a = s["assignments"]
for ai in a:
    print(ai)

#for line in open(infilename):
    #vals = line.split(',')
    #print(len(vals))
