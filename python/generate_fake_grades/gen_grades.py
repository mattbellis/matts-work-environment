import numpy as np
import sys

ngrades = 10
if len(sys.argv)>1:
    ngrades = int(sys.argv[1])

grades = np.random.normal(80,10,ngrades)

int_grades = grades.astype('int')

for g in int_grades:
    print g

