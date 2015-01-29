import sys
import csv 
from pylab import *
import matplotlib.pyplot as plt
from grade_utilities import *
import argparse

import datetime

import plotly
import plotly.plotly as py
from plotly.graph_objs import Box,Layout



################################################################################
# main
################################################################################
def main():

    ############################################################################
    # Parse the arguments
    ############################################################################
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file_name', type=str, default=None, 
            help='Input file name')
    parser.add_argument('--dump-names', dest='dump_names', action='store_true',
            default=True,
            help='Dump the names of the students and an index for each.')
    parser.add_argument('--dump-grades', dest='dump_grades', action='store_true',
            default=False,
            help='Dump the grades for all the students.')
    parser.add_argument('--gvis', dest='gvis', action='store_true',
            default=False,
            help='Dump output for the Google Motion Chart.')
    parser.add_argument('--password', dest='password', default=None, 
            help='Password for mail server.')
    parser.add_argument('--course', dest='course', default=None, 
            help='Course name.')
    parser.add_argument('--student', dest='student', default=None, type=int,
            help='Student name to dump info for or email.')

    args = parser.parse_args()

    ############################################################################

    if args.input_file_name is None:
        print "Must pass in an input file name!"
        parser.print_help()

    filename = args.input_file_name
    infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')

    ############################################################################
    py.sign_in("MatthewBellis", "d6h4et78v5")
    #py = plotly.plotly(username_or_email="MatthewBellis", key="d6h4et78v5")
    ############################################################################

    grade_titles = ["Quizzes", "Homeworks","Exam", "Final exam", "Final grade"]
    student_grades = [[],[],[],[],[]]
    hw_xvals = []
    hw_grades = []
    exam_grades = []
    exam_xvals = []
    quiz_grades = []
    quiz_xvals = []
    final_exam_xvals = []
    final_exam_grades = []

    # Grade weighting
    final_grade_weighting = [0.10,0.25,0.35,0.30]
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

                r = r.strip()

                g = None

                if r=='Quiz' or r=='quiz':
                    g = Grade_file_info("quiz")
                    g.set_grade_index(i)

                elif r=='HW' or r=='hw' or r=='Homework':
                    g = Grade_file_info("hw")
                    g.set_grade_index(i)
                    g.set_add_index(i+1)
                    g.set_subtract_index(i+2)
                            
                elif r=='Exam':
                    g = Grade_file_info("exam")
                    g.set_grade_index(i)

                elif r=='Exam 2' or r=='exam 2':
                    g = Grade_file_info("exam2")
                    g.set_grade_index(i)

                elif r=='Final Exam' or r=='final exam' or r=='Final exam' or r=='Final':
                    g = Grade_file_info("final_exam")
                    g.set_grade_index(i)

                if g is not None:
                    grade_file_infos.append(g)

        if line_num==1:
            for g in grade_file_infos:
                g.set_description(row[g.grade_index])

        if line_num==2:
            for g in grade_file_infos:
                g.set_date(row[g.grade_index])

        # Grab the hw info
        if line_num==3:
            for g in grade_file_infos:
                g.set_max_grade(float(row[g.grade_index]))

                '''
                if g.add_index<len(row) and g.add_index>=0 and row[g.add_index]!='':
                    g.set_add(float(row[g.add_index]))
                if g.subtract_index<len(row) and g.subtract_index>=0 and row[g.subtract_index]!='':
                    g.set_subtract(float(row[g.subtract_index]))
                '''

        '''
        if line_num==3:
            for g in grade_file_infos:
                g.set_internal_index(row[g.grade_index])
        '''

        # Grab the hw info
        #if line_num>=4:
        if line_num>=4:
            ihw = 0 
            row_len = len(row)
            student_name = [row[2],row[3]]
            #email = "z%s@students.niu.edu" % (row[1])
            email = row[1]
            program = row[4]
            major = row[5]
            student_year = row[6]

            cg = Course_grades()

            ################### Grab info #####################
            for g in grade_file_infos:

                score = 0
                is_late = False
                internal_index = -1
                if g.grade_index<row_len and row[g.grade_index]!='':
                    score = float(row[g.grade_index])
                # Check for late grades
                elif g.subtract_index>=0 and g.subtract_index<row_len and row[g.subtract_index]!='':
                    #print student_name[0]
                    #print g.subtract_index
                    #print row[g.subtract_index]
                    score = float(row[g.subtract_index])
                    is_late = True

                #print student_name[0]
                #print is_late
                #print g.internal_index
                grade = Grade(g.grade_type,g.internal_index,score,g.max_grade,g.add,g.subtract,is_late,g.date,g.description)
                #print "%s %3.1f" % (grade.grade_type, grade.score)
                if (grade.grade_type=='hw'):
                    hw_grades.append(grade.grade_pct())
                    #hw_xvals.append(grade.date)
                    hw_xvals.append(datetime.datetime(int(grade.date.split('/')[2]),int(grade.date.split('/')[0]),int(grade.date.split('/')[1]),1))

                if (grade.grade_type=='exam'):
                    exam_grades.append(grade.grade_pct())
                    exam_xvals.append(datetime.datetime(int(grade.date.split('/')[2]),int(grade.date.split('/')[0]),int(grade.date.split('/')[1]),13))

                if (grade.grade_type=='quiz'):
                    quiz_grades.append(grade.grade_pct())
                    quiz_xvals.append(datetime.datetime(int(grade.date.split('/')[2]),int(grade.date.split('/')[0]),int(grade.date.split('/')[1]),23))

                if (grade.grade_type=='final_exam'):
                    final_exam_grades.append(grade.grade_pct())
                    final_exam_xvals.append(datetime.datetime(int(grade.date.split('/')[2]),int(grade.date.split('/')[0]),int(grade.date.split('/')[1])))


                cg.add_grade(grade,grade.grade_type)


            s = Student(student_name,cg,email,program,major,student_year)
            students.append(s)


        line_num += 1

    url = None
    if args.course is not None:

        plotly_title = 'Default grades'
        plotly_filename = 'Default coursename'

        if args.course=='phys120':
            plotly_title = 'PHYS 120 class grades to date.'
            plotly_filename = 'PHYS120_S14'
        elif args.course=='phys260':
            plotly_title = 'PHYS 260 class grades to date.'
            plotly_filename = 'PHYS260_S14'
        elif args.course=='phys110':
            plotly_title = 'PHYS 110 class grades to date.'
            plotly_filename = 'PHYS110_F14'
        elif args.course=='phys310':
            plotly_title = 'PHYS 310 class grades to date.'
            plotly_filename = 'PHYS310_F14'



        #print hw_grades
        #print hw_xvals
        #s={'type':'box' ,'jitter':0.1, 'boxpoints':'all'}
        #s={'type':'box'}
        #axesstyle = {'range':[datetime.datetime(2014,1,15),datetime.datetime(2014,2,15)]}
        #axesstyle = {}
        #l={'title': plotly_title,'xaxis':axesstyle}

        tot_data = []

        #response = py.plot(hw_xvals,hw_grades,style=s,layout=l,filename='grade_example',fileopt='overwrite')
        #data0 = {'y':hw_grades,'x':hw_xvals,'name':"Homework"}
        data0 = Box(y=hw_grades,x=hw_xvals,name="Homework",boxpoints='all',jitter=0.1)
        tot_data.append(data0)
        data1 = Box(y=exam_grades,x=exam_xvals,name="Quizzes",boxpoints='all',jitter=0.1)
        tot_data.append(data1)
        #data1 = {'y':exam_grades,'x':exam_xvals,'name':"Exams"}
        if len(quiz_grades)>0:
            data2 = Box(y=quiz_grades,x=quiz_xvals,name="In-class assignments",boxpoints='all',jitter=0.1)
            #data2 = {'y':quiz_grades,'x':quiz_xvals,'name':"Quizzes"}
            tot_data.append(data2)
        if len(final_exam_grades)>0:
            data3 = Box(y=final_exam_grades,x=final_exam_xvals,name="Final exam",boxpoints='all',jitter=0.1)
            #data3 = {'y':final_exam_grades,'x':final_exam_xvals,'name':"Final exam"}
            tot_data.append(data3)
        #tot_data = [data0]
        #url = py.plot(tot_data,style=s,layout=l,filename=plotly_filename,fileopt='overwrite')
        layout = Layout(title=plotly_title)
        url = py.plot(tot_data,layout=layout,filename=plotly_filename,fileopt='overwrite')

        #url = response['url']
        #filename = response['filename']

        #print response
        print url
        #print filename

    ############################################################################
    # Print out the summary
    ############################################################################
    for i,s in enumerate(students):
        #print s.student_name
        averages, output = s.summary_output(final_grade_weighting)
        #gvis_output = s.gvis_output(final_grade_weighting)
        #nmore_exams = 15 - len(s.grades.exams)
        nmore_exams = 0
        '''
        if len(s.grades.exams)==1:
            nmore_exams = 2
        elif len(s.grades.exams)==2:
            nmore_exams = 1
        elif len(s.grades.exams)==3:
            nmore_exams = 0
        '''

        #nmore_final_exams = 0 # Don't include the final exam
        nmore_final_exams = 1 # Include the final exam in the hypotheticals
        # Uncomment this if you don't want the hypothetical exam grades
        # to be calculated.
        '''
        if len(s.grades.final_exam)==0:
            nmore_final_exams = 1
        '''

        '''
        for j in range(0,nmore_exams):
            dum_grade = Grade('exam',-1,70.0,100.0,0.0,0.0,False,'5/12/2014')
            s.grades.exams.append(dum_grade)
        '''
        for j in range(0,nmore_final_exams):
            dum_grade = Grade('final_exam',-1,70.0,100.0,0.0,0.0,False,'5/12/2014','Test')
            s.grades.final_exam.append(dum_grade)

        #hypothetical_performances = [70.0,80.0,90.0,100.0]
        hypothetical_performances = []
        hypothetical_final_grades = []
        for g in (hypothetical_performances):
            #print g
            '''
            for j in range(0,nmore_exams):
                dum_grade = Grade('exam',-1,g,100.0,0.0,0.0,False,'5/12/2014')
                s.grades.exams[(15-nmore_exams)+j] = dum_grade
            '''
            for j in range(0,nmore_final_exams):
                dum_grade = Grade('final_exam',-1,g,100.0,0.0,0.0,False,'5/12/2014','Test')
                s.grades.final_exam[0] = dum_grade

            dum_averages, dum_output = s.summary_output(final_grade_weighting)
            #print dum_averages
            hypothetical_final_grades.append(dum_averages[-1])

        #print hypothetical_performances 
        #print hypothetical_final_grades 
        ########################################################################
        # For sending out the grades
        ########################################################################
        #subject = "PHYS 283 weekly grade summary"
        subject = "Weekly grade summary"
        msg_body = output

        ########################################################################
        # Tack on the hpotheticals.
        ########################################################################
        #print len(s.grades.final_exam)
        #if 1:
        if len(s.grades.final_exam)==0:
            msg_body += "\n"
            msg_body += "-----------------------------------------\n"
            msg_body += "------- Projected performance -----------\n"
            msg_body += "-----------------------------------------\n"
            msg_body += "If you receive the following grades on all of the remaining exams AND the final exam,\n"
            msg_body += "then you would receive the APPROXIMATELY the final grade for the class ASSUMING you also\n"
            msg_body += "maintain the same average on your quizzes and homeworks.\n"
            msg_body += "\n"
            for pe,pf in zip(hypothetical_performances,hypothetical_final_grades):
                msg_body += "Projected exams/final exams: %5.1f (on each exam)  -  Projected final grade: %5.1f\n" % (pe,pf)


        msg_body += "\nGrades for the class can be found at\n\n\t%s\n" % (url)

        ########################################################################
        # For testing
        ########################################################################
        '''
        subject = "Test of grade summary email system."
        msg_body = "Hi %s %s,\n\n" % (s.student_name[1],s.student_name[0])
        msg_body += "\tI'm testing out this semi-automated system to keep you up to date on your grades.\n"
        msg_body += "\n\tIf I've matched up your name with this email address, "
        msg_body += "could you please let me know in the next class.\n"
        msg_body += "\n\tIf I\'ve made a mistake with the name/email, please let me know that as well.\n"
        msg_body += "\n\tYou don\'t need to email me right now! Just let me know in class.  :)\n"
        msg_body += "\n\tThanks! See you in class!\n\n\nMatt\n\n"
        '''

        #print msg_body
        #print s.email
        if args.dump_names and args.student==None and args.gvis==False:
            print "%d %-20s, %-20s\t%20s" % (i,s.student_name[0],s.student_name[1],s.email)

        if args.dump_grades:
            print s.email
            print msg_body

        if args.student == i:
            print s.email
            print msg_body

        if args.password is not None:
            do_email = False
            if args.student is None:
                do_email = True
            elif args.student == i:
                do_email = True

            if do_email:
                #email_grade_summaries('bellis@slac.stanford.edu','matthew.bellis@gmail.com',subject,msg_body,args.password)
                email_grade_summaries(s.email,'matthew.bellis@gmail.com',subject,msg_body,args.password)

        #print output
        for i,a in enumerate(averages):
            #print averages
            student_grades[i].append(a)

    #exit()

    if args.gvis:
        ret = ""
        ret += "<html>\n"
        ret += "<head>\n"
        ret += '<script type="text/javascript" src="https://www.google.com/jsapi"></script>\n'
        ret += '<script type="text/javascript">\n'
        ret += "google.load('visualization', '1', {'packages':['motionchart']});\n"
        ret += "google.setOnLoadCallback(drawChart);\n"
        ret += "function drawChart() {\n"
        ret += "var data = new google.visualization.DataTable();\n"
        ret += "data.addColumn('string', 'Fruit');\n"
        ret += "data.addColumn('date', 'Date');\n"
        ret += "data.addColumn('string', 'Program');\n"
        ret += "data.addColumn('string', 'Major');\n"
        ret += "data.addColumn('string', 'Student Year');\n"
        ret += "data.addColumn('number', 'Average hw');\n"
        ret += "data.addColumn('number', 'Average exam');\n"
        ret += "data.addColumn('number', 'Average quiz');\n"
        ret += "data.addColumn('number', 'Instantaneous hw');\n"
        ret += "data.addColumn('number', 'Instantaneous exam');\n"
        ret += "data.addColumn('number', 'Instantaneous quiz');\n"
        ret += "data.addColumn('number', 'Final exam');\n"
        ret += "data.addRows([\n"

        for i,s in enumerate(students):
                ret += s.gvis_output(final_grade_weighting)

        ret += "]);\n"
        ret += 'var chart = new google.visualization.MotionChart(document.getElementById(\'chart_div\'));\n'
        ret += 'chart.draw(data, {width:1300, height:800});\n'
        ret += '}\n'
        ret += '</script>\n'
        ret += '</head>\n'
        ret += '<body>\n'
        ret += '<div id="chart_div" style="width:1300px; height: 800px;"></div>\n'
        ret += '</body>\n'
        ret += '</html>\n'

        print ret



    ############################################################################
    # Start plotting the grades.
    ############################################################################
    figs = []
    subplots = []
    #print student_grades
    for k,assignment in enumerate(student_grades):
        figs.append(plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k'))
        subplots.append(figs[k].add_subplot(1,1,1))

        #print grade_titles[k]
        title = "%s" % (grade_titles[k])

        #print assignment
        if len(assignment)>0:
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
                #print hpts
                if len(hpts)>0:
                    hist(hpts,bins=22,facecolor=colors[i],range=(0,110))

            subplots[k].set_xlabel(title, fontsize=14, weight='bold')
            #subplots[k].set_ylim(0,25)

    for i,f in enumerate(figs):
        name = "hw_dist_%d" % (i)
        f.savefig(name)

    #plt.show()





################################################################################
if __name__ == "__main__":
    main()
