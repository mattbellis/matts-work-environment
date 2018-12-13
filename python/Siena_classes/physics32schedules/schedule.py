import matplotlib.pylab as plt
import numpy as np

import sys

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
    eqs = schedule[4]
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
def display_schedule(schedule,schedule_index=3,other_schedule=None):
    terms = schedule[schedule_index]
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
            names = schedule[schedule_index-2]
            credits = schedule[schedule_index-1].astype(int)
            #print("courses")
            #print(courses)
            for idx in indices[0]:
                #print(idx)
                #print(courses[idx],names[idx],credits[idx])
                output += "{0:10s} {1:40s} {2}".format(courses[idx],names[idx],credits[idx])
                if other_schedule is not None:
                    eq = check_equivalency(courses[idx], other_schedule)
                    output += "     {0}".format(eq)
                    if eq.find("None")<0:
                        equivalents.append(eq)
                output += "\n"

            output += "Total credits: {0:38}\n\n".format(sum(credits[indices[0]]))
    equivalents = np.array(equivalents)
    return output,equivalents

################################################################################


infilenames = ['siena_schedule.tsv','rpi_schedule.tsv']

infiles = [open(name,'r') for name in infilenames]

#print(infiles)

siena = parse_file(infiles[0])
other = parse_file(infiles[1])

#print(siena)
#print(other)

###############################################################################
# Display everything
###############################################################################
output,equivalents = display_schedule(siena,other_schedule=other)

print(output)

print()
print("------------------------------------------\nRPI requirements satisfied while at Siena:\n---------------------------------------")
for eq in equivalents:
    print(eq)

print()
print("------------------------------------------\nRPI requirements NOT satisfied while at Siena:\n---------------------------------------")
rpi_classes = other[0].tolist()
for eq in equivalents:
    rpi_classes.remove(eq)
for rpicl in rpi_classes:
    print(rpicl)

output,equivalents = display_schedule(other,schedule_index=2)
print(output)
