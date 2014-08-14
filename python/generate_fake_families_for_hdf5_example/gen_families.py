import numpy as np
import scipy.stats as stats

import sys

nfamilies = int(sys.argv[1])

for i in xrange(nfamilies):

    output = ""

    ############################################################################
    # How many parents?
    ############################################################################
    n = 0
    val = np.random.random()
    if val<0.8:
        n=2
    elif val>=0.8 and val<0.95:
        n=1
    elif val>=0.95:
        n=0

    output += "%d\n" % (n)
    
    for j in range(0,n):
        sex = np.random.choice([0,1])
        gender_identity = np.random.choice([0,1])
        age = np.random.normal(28.0,5.0)
        weight = np.random.normal(100.0,20.0)
        height = np.random.normal(1.7,0.1)
        output += "%d %d %f %f %f\n" % (sex,gender_identity,age,weight,height)
        
    ############################################################################
    # How many children?
    ############################################################################
    n = np.random.choice([0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,5,5,6,6,7,8,9,10,11,12])

    output += "%d\n" % (n)
    
    for j in range(0,n):
        sex = np.random.choice([0,1])
        gender_identity = np.random.choice([0,1])
        age = np.random.normal(28.0,5.0)
        weight = np.random.normal(100.0,20.0)
        height = np.random.normal(1.7,0.1)
        output += "%d %d %f %f %f\n" % (sex,gender_identity,age,weight,height)
        
    ############################################################################
    # How many pets
    ############################################################################
    n = np.random.choice([0,0,1,1,1,1,1,2,2,2,2,3,3,4,4,5])

    output += "%d\n" % (n)
    
    for j in range(0,n):
        animal = np.random.choice([0,0,0,0,0,1,1,1,1,1,2,2,2,3,3,4])
        age = np.random.normal(5.0,1.0)
        output += "%d %f\n" % (animal,age)
        

    ############################################################################
    # Other family information.
    ############################################################################
    for j in range(0,1):
        household_income = np.random.normal(50000.0,10000.0)
        latitude = (45.0-28.0)*np.random.random() + 28.0
        longitude = (120.0-81.0)*np.random.random() + 81.0
        output += "%f %f %f" % (household_income,latitude,longitude)
        

    print output

