import sys
import numpy as np
import openpyxl as op
import shutil
import subprocess as sp

################################################################################
title = "THIS IS OUR TEST"

questions = ['test0000.tex',
             'test0001.tex']

################################################################################


template = open('quiz_template.tex','r')
quiz = open('quiz.tex','w')
solutions = open('solutions.tex','w')

output = ""
for line in template:
    if "%BASIC" in line:
        line = line.replace("%BASIC","")
    output += line
quiz.write(output)
quiz.close()

output = ""
template.seek(0) # Need this to "rewind" the file.
for line in template:
    if "%SOLUTIONS" in line:
        line = line.replace("%SOLUTIONS","")
    output += line
solutions.write(output)
solutions.close()

################################################################################


################################################################################

content = open('content.tex','w')

output = ""
for question in questions:
    output += "\\input{%s}\n" % (question)

content.write(output)
content.close()

################################################################################

wb = op.load_workbook("socrativeQuizTemplate.xlsx")
sh = wb.get_sheet_by_name('Quick Quiz')

sh['B3'] = title


num = 7 # Starting point for Socrativ spreadsheet
options = ["A","B","C","D","E"]
excel_cells = ["C","D","E","F","G"]


for question in questions:
    # Set it to be MC
    cell = "A%d" % (num)
    sh[cell] = "Multiple choice"

    qfile = open(question,'r')
    qtext = ""
    choices = []
    correct_choice = ""
    qbegin = False
    for line in qfile:
        if '\question' in line:
            qtext += ' '.join(line.split()[1:])
            #qtext = qtext.rstrip()
            qbegin = True

        elif qbegin and '\\begin' not in line:
            qtext += line

        elif '\\begin' in line:
            qbegin = False

        elif '\\choice' in line:
            choices.append(' '.join(line.split()[1:]))

        elif '\\CorrectChoice' in line:
            choices.append(' '.join(line.split()[1:]))
            correct_choice = len(choices)-1

    print(qtext)
    print(choices)
    print(correct_choice)
    # The question
    cell = "%s%d" % ("B",num)
    sh[cell] = qtext.rstrip()

    # The options
    for i,choice in enumerate(choices):
        cell = "%s%d" % (excel_cells[i],num)
        sh[cell] = choice.rstrip()

    # The correct answer
    cell = "%s%d" % ("H",num)
    sh[cell] = options[correct_choice]

    num += 1
'''
sh['B7'] = "Who is this?"
sh['C7'] = "Einstein"
sh['D7'] = "Curie"
sh['E7'] = "Newton"
sh['F7'] = "Pauli"
sh['G7'] = "Darwin"
sh['H7'] = "A"
'''

wb.save(filename = "test.xlsx")




################################################################################

################################################################################
sp.run(['make']) #,stdout=sp.PIPE)


