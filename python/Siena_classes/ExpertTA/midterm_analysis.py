import numpy as np
import matplotlib.pylab as plt

import sys

from class_lists import *

infilename = sys.argv[1]

vals = np.loadtxt(infilename,delimiter=',',skiprows=2,dtype=str)

print(vals)

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
                sections[key].append(student_record)

################################################################################
#print(sections['bellis'])
#print(sections['yuksek'])
#print(sections['moustakas'])

def analyze_section(section):

    tot = 0
    newtot = 0

    for entry in section:
        scores = entry[4:-1].astype(float)
        tot += np.mean(scores)

        scores.sort()
        #print(scores)
        #print(scores[2:])
        newmean = np.mean(scores[2:])

        newtot += newmean


        print('{0:20} {1:4.2f}    {2:4.2f}   {3:4.2f}'.format(entry[0], np.mean(scores), newmean, newmean+10))

    print('{0:4.2f}   {1:4.2f}'.format(tot/len(section), newtot/len(section)))




print(len(sections['bellis'][0]))

for key in sections.keys():
    print('{0} -----------------------'.format(key))
    analyze_section(sections[key])

