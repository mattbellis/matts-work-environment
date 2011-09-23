#!/usr/bin/env python

import sys
import csv 
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

################################################################################
def calc_average_of_grades(grades, drop_lowest_score='False'):
    scores = []
    print grades
    for g in grades:
        score = (g.score + g.added - g.subtracted)/g.max_score
        scores.append(score)

    if drop_lowest_score is True:
        scores.sort()
        scores.reverse()
        scores.pop()

    print scores
    return 100*np.mean(scores)


################################################################################
class Grade_file_info:
    def __init__(self, grade_type):
        self.grade_type = grade_type
        self.date = ''
        self.grade_index = -1
        self.max_grade = 100.0
        self.add_index = -1
        self.subtract_index = -1
        self.add = 0.0
        self.subtract = 0.0

    def set_date(self, date):
        self.date = date

    def set_grade_index(self, grade_index):
        self.grade_index = grade_index

    def set_max_grade(self, max_grade):
        self.max_grade = max_grade

    def set_add_index(self, add_index):
        self.add_index = add_index

    def set_subtract_index(self, subtract_index):
        self.subtract_index = subtract_index

    def set_add(self, add):
        self.add = add

    def set_subtract(self, subtract):
        self.subtract = subtract


################################################################################

################################################################################
class Grade:
    def __init__(self, grade_type, score, max_score, added, subtracted, late, date):
        self.grade_type = grade_type
        self.score = score
        self.max_score = max_score
        self.added = added
        self.subtracted = subtracted
        self.late = late
        self.date = date

################################################################################
class Course_grades:
    def __init__(self):
        self.hw = []
        self.quizzes = []
        self.exam1 = []
        self.exam2 = []
        self.final_exam = []

    def add_grade(self, grade, grade_type):
        print "here"
        print grade_type
        if grade_type=='quiz' or grade_type=='Quiz' or grade_type=='Q' or grade_type=='q':
            self.quizzes.append(grade)
        elif grade_type=='HW' or grade_type=='hw' or grade_type=='homework' or grade_type=='Homework':
            self.hw.append(grade)
        elif grade_type=='exam1' or grade_type=='exam_1' or grade_type=='Exam1' or grade_type=='Exam_1':
            self.exam1.append(grade)
        elif grade_type=='exam2' or grade_type=='exam_2' or grade_type=='Exam2' or grade_type=='Exam_2':
            self.exam2.append(grade)
        elif grade_type=='final_exam' or grade_type=='Finalexam' or grade_type=='finalexam' or grade_type=='FinalExam':
            self.final_exam.append(grade)

################################################################################
class Student:
    def __init__(self, student_name, grades):
        self.student_name = student_name
        self.grades = grades

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

    hw_indices = []
    hw_grades = [] 
    hw_max_pts = []
    hw_add = []
    hw_sub = []

    q_indices = []
    e1_indices = []
    e2_indices = []
    fe_indices = []
    names = []


    line_num = 0
    for row in infile:

        # Find out where the quizzes and homeworks are
        if line_num==0:
            for i,r in enumerate(row):

                g = None

                if r=='Quiz' or r=='quiz':
                    q_indices.append(i)
                    g = Grade_file_info("quiz")
                    g.set_grade_index(i)

                elif r=='HW' or r=='hw':
                    hw_indices.append(i)
                    g = Grade_file_info("hw")
                    g.set_grade_index(i)
                    g.set_add_index(i+1)
                    g.set_subtract_index(i+2)
                            
                elif r=='Exam 1' or r=='exam 1':
                    e1_indices.append(i)
                    g = Grade_file_info("exam1")
                    g.set_grade_index(i)

                elif r=='Exam 2' or r=='exam 2':
                    e2_indices.append(i)
                    g = Grade_file_info("exam2")
                    g.set_grade_index(i)

                elif r=='Final Exam' or r=='final exam' or r=='Final exam':
                    fe_indices.append(i)
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

            for index in hw_indices:
                hw_max_pts.append(float(row[index]))

                if index+1>=len(row) or row[index+1]=='':
                    hw_add.append(0.0)
                else:
                    hw_add.append(float(row[index+1]))

                if index+2>=len(row) or row[index+2]=='':
                    hw_sub.append(0.0)
                else:
                    hw_sub.append(float(row[index+2]))

                hw_grades.append([])

        # Grab the hw info
        if line_num>=4:
            row_len = len(row)
            names.append([row[2],row[3]])
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

            for j,index in enumerate(hw_indices):
                norm = hw_max_pts[j]
                #print norm
                grade = 0
                if index<row_len and row[index]!='':
                    grade = 100.0*float(row[index])/norm
                # Check for late grades
                elif index<row_len and row[index+2]!='':
                    grade = 100.0*float(row[index+2])/norm
                    grade -= hw_sub[j]

                grade +=  hw_add[j]

                hw_grades[j].append(grade)
                #print grade

            s = Student(student_name, cg)
            students.append(s)


        line_num += 1

    # Calc the average
    num_hws = len(hw_grades)
    hw_grades.append([])
    num_students = len(hw_grades[0])
    print "num students: %d" % (num_students)
    print "num hws: %d" % (num_hws)
    for j in xrange(num_students):
        avg = 0
        output = "%-15s,%-25s " % (names[j][0],names[j][1])
        for i in xrange(num_hws):
            output += "%7.2f " % (hw_grades[i][j])
            avg += hw_grades[i][j]/float(num_hws)
        hw_grades[-1].append(avg)
        output += "%7.2f " % (avg)
        print output
         


    for s in students:
        print s.student_name
        print calc_average_of_grades(s.grades.hw, 'False')
        for g in s.grades.hw:
            print g.score

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
