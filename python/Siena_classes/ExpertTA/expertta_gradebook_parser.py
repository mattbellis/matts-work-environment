import numpy as np
import matplotlib.pylab as plt

import sys

from class_lists import *

infilename = sys.argv[1]

vals = np.loadtxt(infilename,delimiter=',',skiprows=2,dtype=str)

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
        scores = entry[4:-1]
        output = "{0:12} ".format(entry[0])
        for score in scores[0:20]:
            output += "{0:5s} ".format(score)
        print(output)

#print(len(sections['bellis'][0]))


for i,key in enumerate(sections.keys()):
    print('{0} -----------------------'.format(key))
    analyze_section(sections[key])
