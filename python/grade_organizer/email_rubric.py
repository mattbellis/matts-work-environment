import numpy as np

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

import sys

from grade_utilities import email_grade_summaries

infilename = sys.argv[1]

infile = open(infilename)

students = []

nsections = 2

for i,line in enumerate(infile):

    vals = line.split('\t')

    print(vals)

    if i==0:
        partnames = vals[3:3+nsections]

    elif i==1:
        partpoints = np.array(vals[3:3+nsections]).astype(int)
        tot_available = np.sum(partpoints)

    else:
        name = "%s %s" % (vals[2], vals[1])
        student = {}
        student['name'] = name
        student['email'] = vals[0]
        feedback = ""
        tot = 0
        for j in range(len(partnames)):
            tot += float(vals[j+3])
            feedback += "%s\nPoints: %.2f (out of %d)\n" % (partnames[j], float(vals[j+3]), partpoints[j])
            feedback += "%s\n\n" % (vals[j+3+nsections])
        feedback += "\nTotal points: %.2f\n" % (tot)

        student["feedback"] = feedback

        students.append(student)

#print(students)

for student in students:

    subject = "Breakdown of grading for final project for %s" % (student['name'])
    msg_body =student['feedback']
    email =student['email']
    #email = 'mbellis@siena.edu'
    print(subject)
    print(msg_body)
    if len(sys.argv)>2:
        email_grade_summaries(email,'matthew.bellis@gmail.com',subject,msg_body,sys.argv[2])

    #break








