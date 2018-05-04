import numpy as np

import matplotlib.pylab as plt

import csv
import sys

################################################################################
def average(assignments, drop=0,exclude_high=None):
    scores = [i[1] for i in assignments]
    #print(scores)
    #print(np.mean(scores))
    scores.sort()
    #print(scores)
    avg = -999
    if exclude_high==None:
        avg = np.mean(scores[drop:])
    else:
        avg = np.mean(scores[drop:-exclude_high])

    return avg
################################################################################

infilename = sys.argv[1]
students = []

idx_student = None
if len(sys.argv)>2:
    idx_student = int(sys.argv[2])

with open(infilename, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i,row in enumerate(spamreader):
        #print(len(row))
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
                    #print(a,b,c,d)
                    assignment = {}
                    assignment["name"] = a
                    assignment["date"] = b
                    assignment["maxpoints"] = c
                    assignment["score"] = float(d)
                    assignments.append(assignment)
            student["assignments"] = assignments
            students.append(student)

    #print(points)
#print(students)
min_student = 0
max_student = len(students)
if idx_student is not None:
    min_student = idx_student
    max_student = idx_student + 1

for idx in range(min_student,max_student):
    s = students[idx]
    studentname = s["name"]
    #s = students[0]
    #for key in s.keys():
        #print(s[key])
    a = s["assignments"]
    hws = []
    qus = []
    final = []
    labs = []
    extras = []
    tot = 0
    for ai in a:
        #print(ai)
        m = ai['maxpoints']
        p = ai['score']
        name = ai['name']
        if 'homework' in name.lower() or 'hw' in name.lower():
            #print(p/m)
            hws.append([name,p/m])
        elif 'final exam' in name.lower():
            final = [name,(p+0)/m]
            #print("FINAL: ",final)
        elif 'lab' in name.lower() and 'syllabus' not in name.lower():
            labs.append([name,p/m])
        elif 'quiz' in name.lower() or 'exam' in name.lower():
            #print(p/m)
            tot += p/m
            qus.append([name,p/m])
        else:
            extras.append([name,p/m])

    '''
    for a in hws:
        print(a)
    for a in qus:
        print(a)
    '''

    print("Begin Extras")
    for a in extras:
        print(a)
    print("End Extras")

    for act,label in zip([hws,qus,labs,final],["HOMEWORKS","QUIZZES","LABS","FINAL EXAM"]):
        print('\n----------\n',label,'\n----------')
        if type(act)==list and len(act)>0:
            if type(act[0])==list:
                for a in act:
                    name = a[0]
                    score = a[1]
                    print('{0:.3f}'.format(score),'{0:20s}'.format(name))
            else:
                #print(act)
                print('{0:.3f}'.format(act[1]),'{0:20s}'.format(act[0]))

    print()

    weighting = {}
    activities = {}

    #'''
    # PHYS 260 S16
    weighting["homework"] = 20
    weighting["quizzes"] = 45
    weighting["in-class"] = 10
    weighting["final exam"] = 25

    activities["homework"] = average(hws,drop=1)
    activities["quizzes"] = average(qus,drop=1)
    activities["raw quizzes"] = average(qus,drop=0,exclude_high=2)
    activities["in-class"] = 1.0
    activities["final exam"] = final[-1]
    #'''

    # PHYS 110 S16
    '''
    weighting["homework"] = 25
    weighting["quizzes"] = 35
    weighting["labs"] = 15
    weighting["final exam"] = 25

    activities["homework"] = average(hws) + 0.13
    activities["quizzes"] = average(qus,drop=1)
    activities["raw quizzes"] = average(qus,drop=0,exclude_high=2)
    activities["labs"] = average(labs)
    activities["final exam"] = final[-1] + 0.08
    '''

    #print("QUS")

    tot = 0
    totw = 0
    for key in weighting.keys():
        score = activities[key]
        w = weighting[key]
        print(key,score)
        tot += score*w
        totw += w

    final_grade = tot/totw


    myfmt = '{:.4f}'
    print(myfmt.format(activities['homework']))

    print("summary: ",'{0:20}'.format(studentname),'| hw: ',myfmt.format(activities['homework']),'| quizzes: ',myfmt.format(activities['quizzes']),'| raw quizzes: ',myfmt.format(activities['raw quizzes']),'| final exam: ',myfmt.format(activities['final exam']),'| final grade: ', myfmt.format(final_grade), 'diff (final-rawq): ',myfmt.format(activities['final exam'] - activities['raw quizzes']))


