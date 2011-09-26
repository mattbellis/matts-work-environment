import numpy as np

################################################################################
def calc_average_of_grades(grades, drop_lowest_score='False'):
    scores = []
    #print grades
    for g in grades:
        score = g.grade_pct()
        scores.append(score)

    if drop_lowest_score is True:
        scores.sort()
        scores.reverse()
        scores.pop()

    #print scores
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

    def grade_sum(self):
        ret = self.score + self.added - self.subtracted
        return ret

    def grade_pct(self):
        ret = self.grade_sum()/self.max_score
        return ret

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

