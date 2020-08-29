import sys
import numpy as np

student_list = np.loadtxt('phys015_student_list.dat',delimiter=':',skiprows=0,unpack=True,dtype='str')

vals = np.loadtxt(sys.argv[1],delimiter='\t',skiprows=1,unpack=True,dtype='str')

print(vals)

email = vals[1]
score = vals[2]

print(email)
print(score)

grades = []
names = []
for e,s in zip(email,score):

    g = float(s.split('/')[0])

    idx = student_list[1].tolist().index(e)
    name = student_list[0][idx]
    print('{0:25} {1:25} {2}'.format(e,name,g))

    names.append(name)
    grades.append(g)

print()

grades = np.array(grades)
names = np.array(names)

inds = names.argsort()

for n,g in zip(names[inds], grades[inds]):

    print('{0:25} {1}'.format(n,g))

