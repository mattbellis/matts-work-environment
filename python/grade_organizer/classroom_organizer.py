import numpy as np

import csv
import sys

def average(assignments, drop=0):
    vals = []
    scores = [i[1] for i in assignments]
    #print(scores)
    #print(np.mean(scores))
    scores.sort()
    #print(scores)
    for i in range(0,drop+1):
        vals.append(np.mean(scores[i:]))

    return vals

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
for s in students:
    studentname = s["name"]
    #s = students[0]
    for key in s.keys():
        print(s[key])
    a = s["assignments"]
    hws = []
    qus = []
    extras = []
    tot = 0
    for ai in a:
        print(ai)
        m = ai['maxpoints']
        p = ai['score']
        name = ai['name']
        if 'homework' in name.lower() or 'hw' in name.lower():
            print(p/m)
            hws.append([name,p/m])
        elif 'quiz' in name.lower() or 'exam' in name.lower():
            print(p/m)
            tot += p/m
            qus.append([name,p/m])
        else:
            extras.append([name,p/m])

    for a in hws:
        print(a)
    for a in qus:
        print(a)
    for a in extras:
        print(a)

    weighting = {}
    weighting["homework"] = 20
    weighting["quizzes"] = 45
    weighting["in-class"] = 10
    weighting["final exam"] = 25

    activities = {}
    activities["homework"] = average(hws,drop=1)[-1]
    activities["quizzes"] = average(qus,drop=1)[-1]
    activities["in-class"] = 1.00
    activities["final exam"] = 0.80

    tot = 0
    totw = 0
    for key in activities.keys():
        score = activities[key]
        w = weighting[key]
        print(key,score)
        tot += score*w
        totw += w

    print("summary: ",'{0:20}'.format(studentname),'{:.2f}'.format(activities['homework']),'{:.2f}'.format(activities['quizzes']),'{:.2f}'.format(activities['final exam']),'{:.2f}'.format(tot/totw))


