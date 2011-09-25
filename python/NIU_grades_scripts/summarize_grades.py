#!/usr/bin/env python

import sys
import csv 
from pylab import *
import matplotlib.pyplot as plt
from grade_utilities import *

################################################################################
# main
################################################################################
def main():

    filename = sys.argv[1]
    infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')

    g = Grade('quiz', 10,100,5,10,'True','10/3/11')

    # Grade weighting
    final_grade_weighting = [0.10,0.20,0.20,0.20,0.30]
    # Quizzes
    # HWs
    # Exam 1
    # Exam 2
    # Final Exam

    grade_file_infos = []
    students = []

    line_num = 0
    for row in infile:

        # Find out where the quizzes and homeworks are
        if line_num==0:
            for i,r in enumerate(row):

                g = None

                if r=='Quiz' or r=='quiz':
                    g = Grade_file_info("quiz")
                    g.set_grade_index(i)

                elif r=='HW' or r=='hw':
                    g = Grade_file_info("hw")
                    g.set_grade_index(i)
                    g.set_add_index(i+1)
                    g.set_subtract_index(i+2)
                            
                elif r=='Exam 1' or r=='exam 1':
                    g = Grade_file_info("exam1")
                    g.set_grade_index(i)

                elif r=='Exam 2' or r=='exam 2':
                    g = Grade_file_info("exam2")
                    g.set_grade_index(i)

                elif r=='Final Exam' or r=='final exam' or r=='Final exam':
                    g = Grade_file_info("final")
                    g.set_grade_index(i)

                if g is not None:
                    grade_file_infos.append(g)

        if line_num==1:
            for g in grade_file_infos:
                g.set_date(row[g.grade_index])

        # Grab the hw info
        if line_num==2:
            for g in grade_file_infos:
                g.set_max_grade(float(row[g.grade_index]))

                if g.add_index<len(row) and g.add_index>=0 and row[g.add_index]!='':
                    g.set_add(float(row[g.add_index]))
                if g.subtract_index<len(row) and g.subtract_index>=0 and row[g.subtract_index]!='':
                    g.set_subtract(float(row[g.subtract_index]))

        # Grab the hw info
        if line_num>=4:
            row_len = len(row)
            student_name = [row[2],row[3]]

            cg = Course_grades()

            ################### Grab info #####################
            for g in grade_file_infos:

                score = 0
                is_late = False
                if g.grade_index<row_len and row[g.grade_index]!='':
                    score = float(row[g.grade_index])
                # Check for late grades
                elif g.subtract_index>=0 and g.subtract_index<row_len and row[g.subtract_index]!='':
                    print g.subtract_index
                    print row[g.subtract_index]
                    score = float(row[g.subtract_index])
                    is_late = True

                grade = Grade(g.grade_type,score,g.max_grade,g.add,g.subtract,is_late,g.date)
                print "%s %3.1f" % (grade.grade_type, grade.score)

                cg.add_grade(grade,grade.grade_type)


            s = Student(student_name, cg)
            students.append(s)


        line_num += 1

         
    ############################################################################
    # Print out the summary
    ############################################################################
    for s in students:
        averages = [-1, -1, -1, -1, -1]
        output = "%s %s\n" % (s.student_name[1], s.student_name[0])

        for i,g in enumerate(s.grades.quizzes):
            output +=  "\tQuiz %2d %7.2f/%3.2f\n" % (i,g.grade_sum(),g.max_score)
        avg = calc_average_of_grades(s.grades.quizzes, 'False')
        averages[0] = avg
        output += "\tQuiz avg: %4.2f\n" % (avg)

        for i,g in enumerate(s.grades.hw):
            output +=  "\tHW %2d %7.2f/%3.2f\n" % (i,g.grade_sum(),g.max_score)
        avg = calc_average_of_grades(s.grades.hw, 'False')
        averages[1] = avg
        output += "\tHW avg: %4.2f\n" % (avg)


        tot = 0.0
        tot_wt = 0.0
        for g,w in zip(averages,final_grade_weighting):
            if g>=0:
                tot += g*w
                tot_wt += w
        output += "\tFinal grade: %4.2f\n" % (tot/tot_wt)
            
        print output

    exit()

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
            hist(hpts,bins=22,facecolor=colors[i],range=(0,110))

        subplots[k].set_xlabel(title, fontsize=14, weight='bold')
        subplots[k].set_ylim(0,20)

    for i,f in enumerate(figs):
        name = "hw_dist_%d" % (i)
        f.savefig(name)

    #plt.show()





################################################################################
if __name__ == "__main__":
    main()
