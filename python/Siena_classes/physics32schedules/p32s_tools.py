import matplotlib.pylab as plt
import numpy as np

import sys

import subprocess as sub
import os


################################################################################
def parse_file(schedule_file):
    vals = np.loadtxt(schedule_file,delimiter='\t',skiprows=1,dtype='str')
    schedule_file.seek(0)
    keys = schedule_file.readline()

    return_vals = []
    for val in vals:
        if val[0]!='':
            return_vals.append(val)
    return_vals = np.array(return_vals).transpose()

    return return_vals
################################################################################

################################################################################
def check_equivalency(course,schedule):
    course = course.replace(' ','').upper()
    ret = "None"
    eqs = schedule[5]
    #print(eqs)
    other_courses = schedule[0]
    other_names = schedule[1]
    idx = None
    for i,eq in enumerate(eqs):
        compare = eq.replace(' ','').upper()
        if compare != '':
            #print(course,compare)
            if course.find(compare)>=0:
                idx = i
                #print("FOUND !!!!!!!!!!!!{0}!!!!".format(compare))
    if idx is not None:
        #idx = eqs.tolist().index(course)
        eqcourse = other_courses[idx]
        eqname = other_names[idx]
        #ret = "     {0:10s} {1:40}".format(eqcourse,eqname)
        ret = "{0:10s}".format(eqcourse)

    return ret

################################################################################

################################################################################
def display_schedule(schedule,other_schedule=None,taken_elsewhere=None):
    terms = schedule[3]
    equivalents = []
    #print(terms)
    output = ""
    for term in ['F1', 'S1', 'F2', 'S2', 'F3', 'S3', 'F4', 'S4']:
        #print(term)
        output += "{0} ----------\n".format(term)
        if term in terms:
            indices = np.where(terms==term)
            #print(indices)
            courses = schedule[0]
            names = schedule[1]
            credits = schedule[2].astype(int)
            #print("courses")
            #print(courses)
            for idx in indices[0]:
                #print(idx)
                #print(courses[idx],names[idx],credits[idx])
                output += "{0:20s} {1:40s} {2}".format(courses[idx],names[idx],credits[idx])
                if other_schedule is not None:
                    eq = check_equivalency(courses[idx], other_schedule)
                    output += "     {0}".format(eq)
                    if eq.find("None")<0:
                        equivalents.append(eq.rstrip())
                if taken_elsewhere is not None:
                    if courses[idx] in taken_elsewhere:
                        output += " ************** TAKEN **********"
                output += "\n"

            output += "Total credits: {0:48}\n\n".format(sum(credits[indices[0]]))
    equivalents = np.array(equivalents)
    return output,equivalents

################################################################################
################################################################################
def header():
    h = "\\documentclass{article}\n"
    #h += "\\usepackage{fullpage}\n"
    h += "\\usepackage{ulem}\n"
    h += "\\setlength{\\hoffset}{-1.0in}\n"
    h += "\\setlength{\\textwidth}{7.0in}\n"
    h += "\\setlength{\\topmargin}{-0.5in}\n"
    h += "\\begin{document}\n"

    return h

################################################################################
def footer():
    f = "\\end{document}"

    return f

################################################################################


################################################################################
def latex_schedule(schedule,other_schedule=None,taken_elsewhere=None,school="Siena"):


    #main = "\\noindent \\large{\\bf Siena schedule}\n"
    main = "\\begin{table}\n"
    main += "\\centering\n"
    main += "\\caption{0} {1} schedule{2}\n".format('{',school,'}')
    extra = None
    if taken_elsewhere is None:
        main += "\\begin{tabular}{l l r l} \n"
        extra = " & "
    else:
        main += "\\begin{tabular}{l l r} \n"
        extra = " "
    if taken_elsewhere is None:
        main += "{0:20s} & {1:40s} & {2} & {3}\\\\\n".format("Course","Course name","\# credits","RPI equivalent")
    else:
        main += "{0:20s} & {1:40s} & {2} \\\\\n".format("Course","Course name","\# credits")
    main += "\\hline\n"
    terms = schedule[3]
    equivalents = []
    
    for term in ['F1', 'S1', 'F2', 'S2', 'F3', 'S3', 'F4', 'S4']:
        #output += "{0} ----------\n".format(term)
        main += "\\hline\n"
        if taken_elsewhere is None:
            main += "\\multicolumn{0}{1}{2}{3}\\\\\n".format('{4}','{c}{',term,"}")
        else:
            main += "\\multicolumn{0}{1}{2}{3}\\\\\n".format('{3}','{c}{',term,"}")
        main += "\\hline\n"
        if term in terms:
            indices = np.where(terms==term)
            #print(indices)
            courses = schedule[0]
            names = schedule[1]
            credits = schedule[2].astype(int)
            #print("courses")
            #print(courses)
            for idx in indices[0]:
                #print(idx)
                #print(courses[idx],names[idx],credits[idx])
                taken = False
                if taken_elsewhere is not None:
                    if courses[idx] in taken_elsewhere:
                        taken = True

                if taken==False:
                    main += "{0:20s} & {1:40s} & {2} {3} ".format(courses[idx],names[idx].replace('&','\&'),credits[idx],extra)
                else:
                    main += "{3}{0:20s}{4} & {3}{1:40s}{4} & {2} {5} ".format(courses[idx],names[idx].replace('&','\&'),credits[idx],'\sout{','}',extra)
                if other_schedule is not None:
                    eq = check_equivalency(courses[idx], other_schedule)
                    main += "     {0}".format(eq)
                    if eq.find("None")<0:
                        equivalents.append(eq.rstrip())

                main += "\\\\\n"

            main += "Total credits: & & {0:48} {1}\\\\\n".format(sum(credits[indices[0]]), extra)
    
    main += "\\end{tabular}\n"
    main += "\\end{table}\n"


    return main

################################################################################
def compile_latex(content,filename='myfile'):
    with open('{0}.tex'.format(filename),'w') as f:
        f.write(content)
        f.close()

        commandLine = sub.Popen(['pdflatex', filename])
        commandLine.communicate()

        os.unlink('{0}.aux'.format(filename))
        os.unlink('{0}.log'.format(filename))
        #os.unlink('{0}.tex'.format(filename))


