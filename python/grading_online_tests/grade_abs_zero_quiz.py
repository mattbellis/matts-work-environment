import numpy as np
import sys
import csv


from phys260_info import names,emails
from grade_utilities import email_grade_summaries

filename = sys.argv[1]
infile = csv.reader(open(filename, 'rb'), delimiter='	', quotechar='#')

password = None
if len(sys.argv)>2:
    password = sys.argv[2]

solutions = ['does work to move thermal energy from one region to a warmer region.',
             'weighing a barrel of water before and after it froze.',
             'The linear relationship between the volume and temperature of a gas.',
             'Friction',
             'TRUE',
             'heat and mechanical energy were two forms of the same basic quantity.',
             'there is time for large ice crystals to grow which rupture the cell walls and drain them of their flavor and nutrients.',
             'TRUE',
             'Hydrogen',
             'Dewar',
             'Helium',
             'Mercury',
             'Superfludity',
             'all the atoms wavefunctions overlap and they behave like one atom',
             'Hydrogen',
             'Laser traps., Evaporative cooling.',
             '0.0000001 K',
             '3k by Colorado. 500k by MIT.'
             ]

'''
solutions = ['No',
'Yes',
'Once every week to week and a half',
'Every homework assignment.',
'Every 20-30 minutes',
'The incident being reported to Siena, Failing the course, but you still have the option to appeal',
['20','20%'],
['45','45%'],
['10','10%'],
['25','25%'],
'Yes',
'Yes',
None,
'No',
'Yes',
'If I cite or reference the website.']
'''


student_answers = []
questions = []
linecount = 0
for row in infile:
    student_info = [None,None,None]
    for i,r in enumerate(row):
        if linecount==0:
            print r

    if linecount==0:
        questions = row[2:]
    else:
        student_info[0] = row[1] # Timestamp
        student_info[1] = row[0] # Email
        student_info[2] = row[2:] # Timestamp

        student_answers.append(student_info)

    linecount += 1

#print "#####################################################"
#print student_answers
################################################################################
# Grade the quizzes.
################################################################################

points_per_question = 10
for i,student in enumerate(student_answers):
    total = 0
    total_possible = 0

    print "-------------------------------"
    email = student[0]
    #print email
    name = names[emails.index(email)]
    output = "%-20s %s\n\n\n" % (name,student[0])
    output += "There were %d questions and each was worth %d points\n\n" % (len(solutions),points_per_question)
    #print questions
    #print student[2]
    #print solutions
    for question,answer,solution in zip(questions,student[2],solutions):

        correct = False

        if type(solution) is not list:
            if answer==solution or solution==None:
                correct=True
        else:
            for s in solution:
                if answer==s:
                    correct=True

        output += "-------- \n"
        if correct:
            output += "CORRECT! %s\n\t%s\n\n" % (question,answer)
            total += points_per_question
        else:
            output += "Incorrect.  %s\n\tYou answered:     \t%s\n\tThe correct answer is:\t%s\n\n" % (question,answer,solution)

        total_possible += points_per_question

    output += "\n\n%-25s Grade: %d out of %d ----- %4.2f\n" % (name,total,total_possible,100*total/float(total_possible))

    print output

    if password is not None:
        '''
        if i==0:
            email_grade_summaries('matthew.bellis@gmail.com','matthew.bellis@gmail.com','Thermal Physics - Syllabus quiz grade',output,password)
            exit()
        '''
        email_grade_summaries(email,'matthew.bellis@gmail.com','Thermal Physics - Syllabus quiz grade',output,password)
 

