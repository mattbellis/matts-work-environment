import numpy as np
import matplotlib.pylab as plt

import sys

from class_lists import *

infilename = sys.argv[1]

vals = np.loadtxt(infilename,delimiter=',',skiprows=1,dtype=str)

#print(vals)

sections = {"bellis":[], "yuksek":[], "moustakas":[]}

for key in sections.keys():
    student_listing = None
    if key=='bellis':
        student_listing = bellis
    elif key=='moustakas':
        student_listing = moustakas
    elif key=='yuksek':
        student_listing = yuksek

    idx = []

    for student_record in vals:
        #print(name)
        name = student_record[0]
        for class_names in student_listing:
            #print(class_names,name)
            if name in class_names[0]:
                #print(name)
                #print(student_record)
                sections[key].append(student_record)

################################################################################
#print(sections['bellis'])
#print(sections['yuksek'])
#print(sections['moustakas'])

def analyze_section(section):

    tot = 0
    newtot = 0

    tots = []
    newtots = []

    for entry in section:
        scores = entry[3:-1]
        output = '{0:12s}'.format(entry[0])
        # Average the pre-lecture and checkpoint questions
        for i in range(0,len(scores[0:36]),2):
            avg = np.mean([float(scores[i][:-1]),float(scores[i+1][:-1])])
            output += ' {0:3.0f}'.format(avg)
        print(output)


#print(len(sections['bellis'][0]))

for i,key in enumerate(sections.keys()):
    print('{0} -----------------------'.format(key))
    analyze_section(sections[key])

