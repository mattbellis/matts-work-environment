#!/usr/bin/env python

import sys
import csv 
from pylab import *
import matplotlib.pyplot as plt



################################################################################
# main
################################################################################
def main():

    filename = sys.argv[1]
    infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')

    hw_indices = []
    hw_grades = [] 
    hw_max_pts = []
    q_indices = []
    names = []


    line_num = 0
    for row in infile:

        # Find out where the quizzes and homeworks are
        if line_num==0:
            for i,r in enumerate(row):
                if r=='Quiz' or r=='quiz':
                    q_indices.append(i)
                elif r=='HW' or r=='hw':
                    hw_indices.append(i)

        # Grab the hw info
        if line_num==2:
            for index in hw_indices:
                hw_max_pts.append(float(row[index]))
                hw_grades.append([])

        # Grab the hw info
        if line_num>=4:
            row_len = len(row)
            names.append(row[2])
            for j,index in enumerate(hw_indices):
                norm = hw_max_pts[j]
                #print norm
                grade = 0
                if index<row_len and row[index]!='':
                    grade = 100.0*float(row[index])/norm

                hw_grades[j].append(grade)
                print grade



        line_num += 1

    # Calc the average
    num_hws = len(hw_grades)
    hw_grades.append([])
    num_students = len(hw_grades[0])
    print "num students: %d" % (num_students)
    print "num hws: %d" % (num_hws)
    for j in xrange(num_students):
        avg = 0
        output = "%-25s " % (names[j])
        for i in xrange(num_hws):
            output += "%5.2f " % (hw_grades[i][j])
            avg += hw_grades[i][j]/float(num_hws)
        hw_grades[-1].append(avg)
        output += "%5.2f " % (avg)
        print output
         


    print hw_grades[-1]
    ############################################################################
    # Start parsing the grades.
    ############################################################################
    figs = []
    subplots = []
    for k,assignment in enumerate(hw_grades):
        figs.append(plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k'))
        subplots.append(figs[k].add_subplot(1,1,1))

        title = "HW #%d" % (k+1)
        if k == num_hws:
            title = "Average"


        hw_pts = [[],[],[],[],[]]
        for grade in assignment:
            if grade>=90:
                hw_pts[0].append(grade)
            elif grade>=80 and grade<90:
                hw_pts[1].append(grade)
            elif grade>=70 and grade<80:
                hw_pts[2].append(grade)
            elif grade>=60 and grade<70:
                hw_pts[3].append(grade)
            else:
                hw_pts[4].append(grade)

        colors = ['r','b','g','c','y']
        h = []
        for i,hpts in enumerate(hw_pts):
            hist(hpts,bins=20,facecolor=colors[i],range=(0,100))

        subplots[k].set_xlabel(title, fontsize=14, weight='bold')
        subplots[k].set_ylim(0,20)

    plt.show()





################################################################################
if __name__ == "__main__":
    main()
